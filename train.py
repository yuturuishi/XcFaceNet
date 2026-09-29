import os
import re
import glob
import logging
import platform
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from nets.facenet_training import (get_lr_scheduler, set_optimizer_lr,
                                   triplet_loss, weights_init)
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.logger import setup_logger, fmt_duration
from utils.utils import (get_num_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

#------------------------------------------------------#
#   项目根目录：日志 / checkpoint 等路径均以项目根为基准，
#   在任意目录下执行 train.py 都不会写错位置
#------------------------------------------------------#
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    #------------------------------------------------------#
    #   初始化日志系统：log/train-年月日-时分秒.log
    #------------------------------------------------------#
    _, log_file = setup_logger("train")
    train_start_time = time.time()

    logging.info("=" * 78)
    logging.info("XcFaceNet 人脸识别训练（ArcFace 迭代训练）")
    logging.info("=" * 78)
    logging.info("[启动] 启动时间: %s | 进程 PID: %d",
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.getpid())
    logging.info("[启动] 日志文件: %s", log_file)

    is_use_cuda     = True if torch.cuda.is_available() else False
    seed            = 11
    distributed     = False # 用于指定是否使用单机多卡分布式运行
    sync_bn         = False
    fp16            = True
    train_desc_path = r"D:\datasets\face\ms1mv2\train.txt" # 训练集描述文件（MS1MV2，582万行，2026-09 归拢至 D:\datasets\face 下）
    lfw_path    = r"D:\datasets\face\lfw\lfw" # LFW 评估集图片目录
    lfw_desc_path  = r"D:\datasets\face\lfw\lfw.txt" # LFW 评估集 6000 对描述文件
    lfw_eval_flag = True  # 是否开启LFW评估

    input_shape     = [112, 112, 3] # MS1MV2原生对齐尺寸，减少无效缩放
    backbone        = "mobilenet"   # v2 模型骨干（MobileNetV1，唯一支持的骨干）
    batch_size      = 384             # 384=128个triplet，提升单步计算密度喂满GPU（须为3的倍数）
    Init_Epoch      = 0
    Epoch           = 100             # 默认总训练轮数 100；--resume 续训时从最新 checkpoint 的 epoch 继续训练到该值
    Init_lr             = 5e-4 # 注意：实际lr按 batch/nbs 自适应后会被Adam上限截断为1e-3
    Min_lr              = Init_lr * 0.01 # 模型的最小学习率，默认为最大学习率的0.01
    optimizer_type      = "adam" # 优化器 当使用Adam优化器时建议设置  Init_lr=1e-3，当使用SGD优化器时建议设置   Init_lr=1e-2
    momentum            = 0.9 # 优化器内部使用到的momentum参数
    weight_decay        = 0 # 权值衰减，可防止过拟合
    lr_decay_type       = "cos"
    save_period         = 1 # 多少个epoch保存一次权值，默认每个世代都保存
    save_dir_base      = os.path.join(PROJECT_ROOT, 'checkpoints') # checkpoint 根目录：每次训练自动递增为 checkpoints1/2/3...，避免覆盖历史模型
    # 断点续训（迭代训练）：命令行加 --resume 即从 checkpoints* 全部目录里修改时间最新的 checkpoint
    # 继续训练（续写原目录）；不加参数则默认从头训练（自动递增新目录，永不覆盖历史模型）
    import argparse
    _parser = argparse.ArgumentParser(description='XcFaceNet 训练')
    _parser.add_argument('--resume', action='store_true', help='从 checkpoints* 最新 checkpoint 继续迭代训练')
    _args, _ = _parser.parse_known_args()
    resume_training     = _args.resume
    samples_per_epoch   = 600000       # 每epoch采样量（在线随机triplet，无需扫完全部5.8M；种子bug修复后数据不再重复，加倍以喂饱85k分类头）

    # 数据加载多进程：24 逻辑核，主进程喂数据是之前 GPU 吃不满的根源
    # （每 epoch 4500+ 步全部串行读图+增广，GPU 每步都在空等）。
    # 8 worker 并行预取 + persistent_workers 避免每 epoch 重新拉起进程；
    # 历史「Windows 验证阶段多进程 OOM」与机器内存无关，是当时内存不足导致，现机器内存充足。
    # 8→6 worker：Windows spawn 会把整个 Dataset（~580万条路径）pickle 给每个 worker，
    # 8 worker 拉起瞬时内存曾触发 MemoryError（2026-09-16 实跑），6 为安全值；速度同样是 GPU 打满。
    # worker 持久化 + 预取后 GPU 利用率从个位数拉满（实测 0.77→0.26 秒/步，约 3 倍提速）。
    num_workers     = 6
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    #------------------------------------------------------#
    #   运行环境信息
    #------------------------------------------------------#
    logging.info("[环境] 操作系统: %s | Python: %s", platform.platform(), platform.python_version())
    logging.info("[环境] PyTorch: %s | CUDA 可用: %s | GPU 数量: %d",
                 torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
    if torch.cuda.is_available():
        _props = torch.cuda.get_device_properties(0)
        logging.info("[环境] GPU: %s | 显存: %.1f GB | 计算能力: %d.%d",
                     torch.cuda.get_device_name(0), _props.total_memory / 1024 ** 3,
                     _props.major, _props.minor)
    logging.info("[环境] 混合精度(fp16): %s | 随机种子: %d | 分布式: %s | num_workers: %d",
                 fp16, seed, distributed, num_workers)

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            logging.info("[rank = %d, local_rank = %d] training... Gpu Device Count : %d",
                         rank, local_rank, ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0
    logging.info("[环境] 训练设备: %s", device)

    #------------------------------------------------------#
    #   数据集统计（含耗时）
    #------------------------------------------------------#
    t0 = time.time()
    num_classes = get_num_classes(train_desc_path)
    logging.info("[数据集] 训练集: MS1MV2 | 描述文件: %s", train_desc_path)
    logging.info("[数据集] 身份数(类别数)统计完成: %d 类 | 扫描描述文件耗时: %s",
                 num_classes, fmt_duration(time.time() - t0))

    def find_latest_checkpoint(save_dir):
        patterns = [
            os.path.join(save_dir, "model_epoch*_train_loss*.pth"),
            os.path.join(save_dir, "model_*_loss*.pth"),
        ]
        checkpoints = []
        for pattern in patterns:
            checkpoints.extend(glob.glob(pattern))
        if not checkpoints:
            return None, 0
        latest = max(checkpoints, key=os.path.getmtime)
        match = re.search(r"model_epoch(\d+)_", os.path.basename(latest))
        if not match:
            match = re.search(r"model_(\d+)_", os.path.basename(latest))
        last_epoch = int(match.group(1)) if match else 0
        return latest, last_epoch

    #------------------------------------------------------#
    #   训练输出目录自动递增：
    #   checkpoints 不存在 → 直接用 checkpoints；
    #   已存在 → 依次检查 checkpoints1/2/3... 取第一个空缺号。
    #   每次训练独立目录，历史模型永不覆盖。
    #------------------------------------------------------#
    def next_save_dir(base):
        if not os.path.isdir(base):
            return base
        i = 1
        while os.path.isdir("%s%d" % (base, i)):
            i += 1
        return "%s%d" % (base, i)

    def find_latest_checkpoint_any(base):
        """在 checkpoints* 全部目录中搜索修改时间最新的 checkpoint（断点续训用）"""
        candidates = [base] if os.path.isdir(base) else []
        i = 1
        while os.path.isdir("%s%d" % (base, i)):
            candidates.append("%s%d" % (base, i))
            i += 1
        latest, latest_dir, last_epoch = None, None, 0
        for d in candidates:
            path, ep = find_latest_checkpoint(d)
            if path and (latest is None or os.path.getmtime(path) > os.path.getmtime(latest)):
                latest, latest_dir, last_epoch = path, d, ep
        return latest, latest_dir, last_epoch

    # 目录决策：
    #   resume_training=True  → 搜索 checkpoints* 最新 checkpoint 断点续训，续写原目录；
    #   resume_training=False → 自动递增出新目录（checkpoints1/2/3...），从头训练，避免模型覆盖。
    pre_model_path = None
    if resume_training:
        resume_path, resume_dir, resume_epoch = find_latest_checkpoint_any(save_dir_base)
        if resume_path:
            save_dir = resume_dir
            Init_Epoch = resume_epoch
            pre_model_path = resume_path
            logging.info("[续训] 检测到 checkpoint，从 epoch %03d 继续: %s（续写目录 %s）",
                         Init_Epoch, resume_path, save_dir)
        else:
            save_dir = next_save_dir(save_dir_base)
            # 修复：自增目录必须显式创建，否则首次 torch.save 必然 FileNotFoundError
            os.makedirs(save_dir, exist_ok=True)
            logging.warning("[续训] checkpoints* 目录均无 checkpoint，将从头训练，输出目录: %s", save_dir)
    else:
        save_dir = next_save_dir(save_dir_base)
        os.makedirs(save_dir, exist_ok=True)
        logging.info("[输出] 本次训练 checkpoint 输出目录: %s（历史目录不覆盖）", save_dir)

    #---------------------------------#
    #   载入模型并加载预训练权重
    #---------------------------------#
    model = Facenet(backbone=backbone, num_classes=num_classes)

    # 网络结构信息
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("[网络] Backbone: %s | 输入尺寸: %dx%dx%d | 嵌入维度: 128 维（L2归一化输出）",
                 backbone, input_shape[0], input_shape[1], input_shape[2])
    logging.info("[网络] 损失函数: 在线TripletLoss + ArcFace(s=%.1f, m=%.2f, 无bias, 权重归一化)",
                 model.s, model.m)
    logging.info("[网络] 分类头: %d 类 | 参数量: 总计 %.2fM（%.1f MB, fp32）| 可训练 %.2fM",
                 num_classes, total_params / 1e6, total_params * 4 / 1024 / 1024, trainable_params / 1e6)
    logging.debug("[网络] 完整结构:\n%s", model)

    logging.info("[设备] is_use_cuda: %s | device: %s", is_use_cuda, device)
    if pre_model_path:
        logging.info("[权重] 加载 checkpoint: %s", pre_model_path)

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(pre_model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            logging.info("[权重] 成功载入 %d 个参数张量 | 未载入 %d 个（head部分未载入属正常，backbone未载入属错误）",
                         len(load_key), len(no_load_key))
            if no_load_key:
                logging.debug("[权重] 未载入 Key 列表: %s", str(no_load_key))

    loss            = triplet_loss()
    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        loss_history = LossHistory()
    else:
        loss_history = None

    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        logging.warning("Sync_bn is not support in one gpu or not distributed.")

    if is_use_cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------------#
    #   LFW估计
    #---------------------------------#
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_path, pairs_path=lfw_desc_path, image_size=input_shape), batch_size=32, shuffle=False,
        num_workers=2, persistent_workers=True) if lfw_eval_flag else None

    #-------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    #-------------------------------------------------------#
    val_split = 0.01
    t0 = time.time()
    with open(train_desc_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    logging.info("[数据集] 描述文件读取+打乱耗时: %s", fmt_duration(time.time() - t0))
    logging.info("[数据集] 总样本 %d 条 | 训练集 %d 条 | 验证集 %d 条（val_split=%.0f%%）",
                 len(lines), num_train, num_val, val_split * 100)
    logging.info("[数据集] LFW 评估集: %s（6000 对，10 折交叉验证，每epoch评估一次）", lfw_desc_path)

    show_config(
        num_classes = num_classes, backbone = backbone, model_path = pre_model_path or "（随机初始化）", input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val, \
        samples_per_epoch = samples_per_epoch, lfw_eval_flag = lfw_eval_flag, fp16 = fp16
    )

    if True:
        if batch_size % 3 != 0:
            raise ValueError("batch_size must be the multiple of 3.")
        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        logging.info("[学习率] 自适应调整后: Init_lr=%.3e（限幅[%.1e, %.1e]）| Min_lr=%.3e | 衰减方式: %s",
                     Init_lr_fit, lr_limit_min, lr_limit_max, Min_lr_fit, lr_decay_type)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        #---------------------------------------#
        #   构建数据集加载器。
        #---------------------------------------#
        t0 = time.time()
        train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_classes, random = True, samples_per_epoch=samples_per_epoch)
        val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_classes, random = False, samples_per_epoch=max(samples_per_epoch // 20, num_classes))
        logging.info("[数据集] Dataset 对象构建耗时: %s", fmt_duration(time.time() - t0))

        #---------------------------------------#
        #   判断每一个世代的长度
        #   DataLoader.batch_size = batch_size//3（每个item是一个triplet）
        #---------------------------------------#
        epoch_step      = len(train_dataset) // (batch_size // 3)
        epoch_step_val  = len(val_dataset) // (batch_size // 3)
        logging.info("[数据集] 每epoch采样: 训练 %d 个triplet（%d 步）| 验证 %d 个triplet（%d 步）",
                     len(train_dataset), epoch_step, len(val_dataset), epoch_step_val)
        logging.info("[数据集] 每步喂入 %d 张图（%d 个triplet x 3）| 每epoch实际训练 %d 张图",
                     batch_size, batch_size // 3, epoch_step * batch_size)

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        loader_kwargs = dict(
            shuffle=shuffle,
            batch_size=batch_size // 3,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset_collate,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4  # 控制预取内存峰值（原6偏高）

        gen = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
        gen_val = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

        #------------------------------------------------------#
        #   开始逐 epoch 训练
        #   MemoryError（多为 dataloader worker 拉起瞬时内存）→ 重试同一 epoch；
        #   其他异常仍跳过该轮（避免死循环卡死训练）。
        #------------------------------------------------------#
        epoch = Init_Epoch
        fail_tries = 0
        while epoch < Epoch:
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            try:
                fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, is_use_cuda, LFW_loader, batch_size//3, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank)
                epoch += 1
                fail_tries = 0
            except MemoryError:
                fail_tries += 1
                logging.exception("[Epoch %03d] 内存不足（多为 dataloader worker 拉起），清理缓存后重试（第 %d 次）",
                                  epoch + 1, fail_tries)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if fail_tries >= 3:
                    logging.error("[Epoch %03d] 连续 %d 次内存失败，跳过该 epoch", epoch + 1, fail_tries)
                    epoch += 1
                    fail_tries = 0
            except Exception as e:
                logging.exception("[Epoch %03d] 训练异常，跳过本轮回继续: %s", epoch + 1, e)
                # 清理GPU缓存避免显存泄漏
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                epoch += 1

        #------------------------------------------------------#
        #   训练结束总结
        #------------------------------------------------------#
        total_time = time.time() - train_start_time
        logging.info("=" * 78)
        logging.info("训练流程结束")
        logging.info("[总结] 计划 epoch: %d（从 %d 开始）| 实际完成: %d 个",
                     Epoch, Init_Epoch, len(loss_history.losses) if loss_history else 0)
        logging.info("[总结] 本次训练总耗时: %s | 平均每个epoch: %s",
                     fmt_duration(total_time), fmt_duration(total_time / max(Epoch - Init_Epoch, 1)))
        if loss_history is not None and loss_history.best_epoch > 0:
            logging.info("[总结] 历史最优精度: %.5f（epoch %03d）", loss_history.best_acc, loss_history.best_epoch)
        logging.info("[总结] checkpoint 目录: %s", os.path.abspath(save_dir))
        logging.info("[总结] 日志文件: %s", log_file)
        logging.info("=" * 78)
