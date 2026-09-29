import logging
import random
import numpy as np
import torch
import cv2
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        # image.show()
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        image   = image.resize((nw,nh), Image.BICUBIC)
        # image.show()
        new_image = Image.new('RGB', size, (128,128,128))
        # new_image.show()
        wss = (w-nw)//2
        hss = (h-nh)//2
        new_image.paste(image, (wss, hss))
        # new_image.show()
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 不再强制 cudnn.deterministic/benchmark（与训练脚本 benchmark=True 自相矛盾）：
    # 随机triplet采样本就无法严格复现，训练脚本按需自行设置 benchmark 以启用最快卷积算法

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    # 修复：必须把 worker_id 混入种子。旧实现 rank+seed 在单卡下对所有 worker 相同，
    # 配合"忽略index的随机triplet采样"导致 6 个 worker 产出完全相同的采样流，
    # 每 epoch 实际唯一数据只有名义的 1/6（ArcFace 分类头精度恒 0% 的根因）。
    worker_seed = seed + worker_id + rank * 1000
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # 每个 worker 关闭 cv2 内部多线程：避免 N workers × 全核线程互相争抢 CPU（PyTorch 官方建议做法）
    cv2.setNumThreads(0)

def preprocess_input(image):
    image /= 255.0 
    return image

def show_config(**kwargs):
    logging.info("Configurations:")
    logging.info("-" * 70)
    logging.info("|%25s | %40s|" % ('keys', 'values'))
    logging.info("-" * 70)
    for key, value in kwargs.items():
        logging.info("|%25s | %40s|" % (str(key), str(value)))
    logging.info("-" * 70)