import glob
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

from utils.logger import fmt_duration
from utils.utils import get_lr
from utils.utils_metrics import evaluate

# 训练/验证过程中每隔多少步输出一次进度日志
LOG_EVERY = 500


def fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, test_loader, Batch_size, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank):
    epoch_t0 = time.time()
    total_triple_loss   = 0
    total_CE_loss       = 0
    total_accuracy      = 0

    val_total_triple_loss   = 0
    val_total_CE_loss       = 0
    val_total_accuracy      = 0

    # ArcFace 已在 model.forward 中做归一化+角度间隔，输出为缩放后的 logits
    # 因此这里用标准 CrossEntropyLoss（内部已含 log_softmax + NLL），无需再 log_softmax
    arcface_ce = nn.CrossEntropyLoss()

    model_train.train()
    if local_rank == 0:
        logging.info("========== Epoch %03d/%03d 开始 ==========", epoch + 1, Epoch)
        logging.info("[训练] 本epoch数据规模: %d 步 x %d 图/步 = %d 图 | 当前学习率: %.3e",
                     epoch_step, Batch_size * 3, epoch_step * Batch_size * 3, get_lr(optimizer))

    #------------------------------------------------------#
    #   训练阶段
    #------------------------------------------------------#
    train_t0 = time.time()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank, non_blocking=True)
                labels  = labels.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad()
        if not fp16:
            outputs1, outputs2 = model_train(images, "train", labels)

            _triplet_loss   = loss(outputs1, Batch_size)
            # ArcFace loss：outputs2 已是 s*cos(theta+m)，直接用 CE
            _CE_loss        = arcface_ce(outputs2, labels)
            _loss           = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs1, outputs2 = model_train(images, "train", labels)

                _triplet_loss   = loss(outputs1, Batch_size)
                _CE_loss        = arcface_ce(outputs2, labels)
                _loss           = _triplet_loss + _CE_loss
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            accuracy         = torch.mean((torch.argmax(outputs2, dim=-1) == labels).type(torch.FloatTensor))

        total_triple_loss   += _triplet_loss.item()
        total_CE_loss       += _CE_loss.item()
        total_accuracy      += accuracy.item()

        if local_rank == 0 and (iteration + 1) % LOG_EVERY == 0:
            elapsed  = time.time() - train_t0
            speed    = elapsed / (iteration + 1)
            remain_s = speed * (epoch_step - iteration - 1)
            logging.info(
                "[训练] Epoch %03d 进度 %d/%d (%.1f%%) | triplet=%.4f CE=%.4f loss=%.4f | lr=%.3e | %.2f秒/步 | 预计剩余 %s",
                epoch + 1, iteration + 1, epoch_step, 100.0 * (iteration + 1) / epoch_step,
                total_triple_loss / (iteration + 1), total_CE_loss / (iteration + 1),
                (total_triple_loss + total_CE_loss) / (iteration + 1),
                get_lr(optimizer), speed, fmt_duration(remain_s))

    train_time = time.time() - train_t0
    if local_rank == 0:
        logging.info("[训练] Epoch %03d 训练阶段完成 | 耗时 %s | 平均 %.2f 秒/步",
                     epoch + 1, fmt_duration(train_time), train_time / max(epoch_step, 1))

    #------------------------------------------------------#
    #   验证阶段
    #------------------------------------------------------#
    val_t0 = time.time()
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank, non_blocking=True)
                labels  = labels.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            outputs1, outputs2 = model_train(images, "train", labels)

            _triplet_loss   = loss(outputs1, Batch_size)
            _CE_loss        = arcface_ce(outputs2, labels)
            _loss           = _triplet_loss + _CE_loss

            accuracy        = torch.mean((torch.argmax(outputs2, dim=-1) == labels).type(torch.FloatTensor))

            val_total_triple_loss   += _triplet_loss.item()
            val_total_CE_loss       += _CE_loss.item()
            val_total_accuracy      += accuracy.item()

        if local_rank == 0 and (iteration + 1) % LOG_EVERY == 0:
            logging.info(
                "[验证] Epoch %03d 进度 %d/%d (%.1f%%) | val_triplet=%.4f val_CE=%.4f val_loss=%.4f",
                epoch + 1, iteration + 1, epoch_step_val, 100.0 * (iteration + 1) / epoch_step_val,
                val_total_triple_loss / (iteration + 1), val_total_CE_loss / (iteration + 1),
                (val_total_triple_loss + val_total_CE_loss) / (iteration + 1))

    val_time = time.time() - val_t0

    #------------------------------------------------------#
    #   关键修复：先保存 checkpoint，再做 LFW 评估
    #   此前 LFW 评估崩溃会导致整个 epoch 训练成果丢失
    #------------------------------------------------------#
    saved_pre_lfw = None
    if local_rank == 0:
        logging.info("[验证] Epoch %03d 验证阶段完成 | 耗时 %s", epoch + 1, fmt_duration(val_time))

        train_loss = (total_triple_loss + total_CE_loss) / epoch_step
        val_loss = (val_total_triple_loss + val_total_CE_loss) / epoch_step_val
        train_acc = total_accuracy / epoch_step
        val_acc = val_total_accuracy / epoch_step_val

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            # 清理之前 epoch 遗留的临时文件，避免堆积占磁盘
            for stale in glob.glob(os.path.join(save_dir, '*_lfw_evaluating.*')):
                try:
                    os.remove(stale)
                except OSError:
                    pass
            # 临时文件名（LFW 评估完成后重命名为含 lfw_acc 的正式名）
            save_name = 'model_epoch%03d_train_loss%.4f_val_loss%.4f_lfw_evaluating' % (
                epoch + 1, train_loss, val_loss)
            model_pth_filepath = os.path.join(save_dir, save_name + ".pth")
            # 只存 state_dict(.pth)：导出 onnx / 断点续训 / 测试均用 .pth，不再双份存整模型(.pt)
            torch.save(model.state_dict(), model_pth_filepath)
            saved_pre_lfw = model_pth_filepath
            logging.info("[保存] checkpoint 已保存（LFW评估前临时名）: %s", model_pth_filepath)

    lfw_acc = 0.0
    lfw_acc_std = 0.0
    if lfw_eval_flag:
        lfw_t0 = time.time()
        logging.info("[LFW评估] 开始 LFW 验证（6000 对，10 折交叉验证）")
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(test_loader):
            with torch.no_grad():
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                if cuda:
                    data_a, data_p = data_a.cuda(local_rank), data_p.cuda(local_rank)
                out_a, out_p = model_train(data_a), model_train(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels      = np.array([sublabel for label in labels for sublabel in label])
        distances   = np.array([subdist for dist in distances for subdist in dist])
        try:
            _, _, accuracy, _, _, _, _ = evaluate(distances, labels)
        except Exception as e:
            logging.error("[LFW评估] 评估失败: %s，本次精度记为 0", e)
            accuracy = np.zeros(10)
        lfw_acc = float(np.mean(accuracy))
        lfw_acc_std = float(np.std(accuracy))
        # 释放 LFW 评估占用的缓存，缓解长时训练内存压力
        del labels, distances
        if cuda:
            torch.cuda.empty_cache()
        logging.info("[LFW评估] 完成 | 耗时 %s | LFW_Accuracy: %2.5f +- %2.5f",
                     fmt_duration(time.time() - lfw_t0), lfw_acc, lfw_acc_std)

    if local_rank == 0:
        # 分类头 top-1 精度：85742 类从头训练极难脱离 0.00%，恒为 0 时无参考价值不打印；
        # 一旦非零（≥0.01%）说明分类头开始记忆身份，届时自动补打。嵌入质量以 LFW 为准。
        acc_note = ""
        if train_acc >= 0.0001 or val_acc >= 0.0001:
            acc_note = " | train_acc=%.2f%% val_acc=%.2f%%" % (train_acc * 100, val_acc * 100)
        if lfw_eval_flag:
            logging.info("[汇总] Epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | LFW_acc=%.5f%s",
                         epoch + 1, Epoch, train_loss, val_loss, lfw_acc, acc_note)
        else:
            logging.info("[汇总] Epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | train_acc=%.2f%% | val_acc=%.2f%%（未开启LFW评估）",
                         epoch + 1, Epoch, train_loss, val_loss, train_acc * 100, val_acc * 100)

        loss_history.append_loss(epoch, lfw_acc if lfw_eval_flag else train_acc, train_loss, val_loss)

        if saved_pre_lfw is not None:
            # LFW 评估已完成，将临时文件重命名为含 lfw_acc 的正式文件名
            final_name = 'model_epoch%03d_train_loss%.4f_val_loss%.4f_lfw_acc%.4f' % (
                epoch + 1, train_loss, val_loss, lfw_acc)
            final_pth = os.path.join(save_dir, final_name + ".pth")
            try:
                os.rename(saved_pre_lfw, final_pth)
                logging.info("[保存] 正式模型已保存: %s", final_pth)
            except OSError:
                # 重命名失败时保留临时文件（仍可被断点续训识别）
                logging.warning("[保存] 重命名失败，保留临时文件: %s", saved_pre_lfw)

        # best 另存：LFW 创新高时额外复制一份，避免从几十个权重里人工挑最优
        if lfw_eval_flag and lfw_acc > 0 and saved_pre_lfw is not None:
            best_marker = os.path.join(save_dir, '_best_lfw.txt')
            prev_best = 0.0
            if os.path.exists(best_marker):
                try:
                    prev_best = float(open(best_marker).read().strip())
                except (ValueError, OSError):
                    prev_best = 0.0
            if lfw_acc > prev_best:
                best_name = 'model_best_lfw%.4f_ep%03d' % (lfw_acc, epoch + 1)
                best_pth = os.path.join(save_dir, best_name + '.pth')
                try:
                    import shutil
                    shutil.copyfile(final_pth if os.path.exists(final_pth) else saved_pre_lfw, best_pth)
                    with open(best_marker, 'w') as f:
                        f.write('%.5f' % lfw_acc)
                    # 清理上一份 best（只保留最新一份，避免占盘）
                    import glob as _glob, re as _re
                    for old in _glob.glob(os.path.join(save_dir, 'model_best_lfw*.pth')):
                        if old != best_pth and _re.search(r'lfw([\d.]+)_ep(\d+)', old) and \
                           float(_re.search(r'lfw([\d.]+)', old).group(1)) < lfw_acc:
                            try:
                                os.remove(old)
                            except OSError:
                                pass
                    logging.info("[best] ★ LFW 新高 %.5f（原 %.5f），best 权重已另存: %s",
                                 lfw_acc, prev_best, best_pth)
                except OSError as e:
                    logging.warning("[best] best 权重另存失败: %s", e)

        logging.info("========== Epoch %03d/%03d 结束 | 本epoch总耗时 %s ==========",
                     epoch + 1, Epoch, fmt_duration(time.time() - epoch_t0))
