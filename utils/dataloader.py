import os
import random
import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data.dataset import Dataset
from .utils import cvtColor, preprocess_input, resize_image

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class FacenetDataset(Dataset):
    def __init__(self, input_shape, lines, num_classes, random, samples_per_epoch=None):
        self.input_shape    = input_shape
        self.lines          = lines
        self.num_classes    = num_classes
        self.do_augment     = random
        self.h              = int(input_shape[0])
        self.w              = int(input_shape[1])
        
        #------------------------------------#
        #   路径和标签
        #------------------------------------#
        self.paths_by_class = [[] for _ in range(num_classes)]
        self.classes_ge2 = []
        self.classes_ge1 = []

        self.load_dataset()
        total = sum(len(v) for v in self.paths_by_class)
        # Triplet采样不依赖index；限制每epoch样本量，避免空转索引5.8M条
        if samples_per_epoch is None:
            self.length = total
        else:
            self.length = max(int(samples_per_epoch), self.num_classes)
        
    def __len__(self):
        return self.length

    def _sim_det_crop(self, image):
        """模拟部署端「YOLO 紧脸框 + 外扩裁剪」的输入口径。

        部署时模型吃的是检测框(眉上~下巴)外扩后的正方形 crop，与训练用的
        五点对齐脸口径不一致，导致部署同人得分偏低。此增强在训练时随机
        生成各种紧框/外扩/平移/灰边的 crop，让模型对部署口径鲁棒。
        输入: BGR 对齐脸(HxW)；输出: 同尺寸 BGR。
        """
        h, w = image.shape[:2]
        eyes_y = 0.46 * h                       # 对齐脸模板：眼位约在 0.46H
        bh = rand(0.45, 0.70) * h               # 模拟紧框高度（眉上~下巴）
        bw = bh * rand(0.80, 0.95)              # 紧框宽高比
        eyes_in_box = rand(0.36, 0.42) * bh     # 眼在紧框内的相对高度
        top = eyes_y - eyes_in_box
        cx = 0.5 * w + rand(-0.04, 0.04) * w
        cy = top + bh / 2.0
        side = max(bw, bh) * rand(1.0, 1.9)     # 部署端外扩倍数分布
        ax1 = cx - side / 2.0 + rand(-0.05, 0.05) * side
        ay1 = cy - side / 2.0 + rand(-0.05, 0.05) * side
        ax2, ay2 = ax1 + side, ay1 + side
        pl, pt = int(max(0, -ax1)), int(max(0, -ay1))
        pr, pb = int(max(0, ax2 - w)), int(max(0, ay2 - h))
        x1, y1 = int(max(0, ax1)), int(max(0, ay1))
        x2, y2 = int(min(w, ax2)), int(min(h, ay2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return image
        patch = image[y1:y2, x1:x2]
        if pl or pt or pr or pb:
            mode = cv2.BORDER_CONSTANT if np.random.rand() < 0.5 else cv2.BORDER_REPLICATE
            patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, mode, value=(128, 128, 128))
        return cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)

    def _load_face(self, path):
        if _HAS_CV2:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            if self.do_augment and random.random() < 0.5:
                image = cv2.flip(image, 1)
            if self.do_augment and random.random() < 0.5:
                image = self._sim_det_crop(image)
            if image.shape[0] != self.h or image.shape[1] != self.w:
                image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) * (1.0 / 255.0)
            return np.transpose(image, (2, 0, 1))

        image = cvtColor(Image.open(path).convert('RGB'))
        if self.do_augment and self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 已对齐人脸直接resize，不做letterbox，显著减少CPU开销
        image = resize_image(image, [self.w, self.h], letterbox_image=False)
        image = preprocess_input(np.array(image, dtype='float32'))
        return np.transpose(image, [2, 0, 1])

    def __getitem__(self, index):
        images = np.empty((3, 3, self.h, self.w), dtype=np.float32)
        labels = np.empty((3,), dtype=np.int64)

        for _ in range(20):
            try:
                c = self.classes_ge2[random.randrange(len(self.classes_ge2))]
                selected_path = self.paths_by_class[c]
                i1, i2 = random.sample(range(len(selected_path)), 2)
                images[0] = self._load_face(selected_path[i1])
                images[1] = self._load_face(selected_path[i2])
                labels[0] = c
                labels[1] = c

                current_c = c
                while current_c == c:
                    current_c = self.classes_ge1[random.randrange(len(self.classes_ge1))]
                selected_path = self.paths_by_class[current_c]
                images[2] = self._load_face(selected_path[random.randrange(len(selected_path))])
                labels[2] = current_c
                return images, labels
            except (FileNotFoundError, OSError):
                continue

        return images, labels

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            img_path = path_split[1].split()[0]
            label = int(path_split[0])
            if 0 <= label < self.num_classes:
                self.paths_by_class[label].append(img_path)

        self.classes_ge1 = [cid for cid, paths in enumerate(self.paths_by_class) if len(paths) >= 1]
        self.classes_ge2 = [cid for cid, paths in enumerate(self.paths_by_class) if len(paths) >= 2]
        if len(self.classes_ge2) == 0:
            raise ValueError("No identity has at least 2 images for triplet sampling.")

# DataLoader中collate_fn使用
def dataset_collate(batch):
    images = np.stack([item[0] for item in batch], axis=0)  # [B, 3, 3, H, W]
    labels = np.stack([item[1] for item in batch], axis=0)  # [B, 3]
    images = np.concatenate([images[:, 0], images[:, 1], images[:, 2]], axis=0)
    labels = np.concatenate([labels[:, 0], labels[:, 1], labels[:, 2]], axis=0)
    return torch.from_numpy(images), torch.from_numpy(labels)

class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(LFWDataset, self).__init__(dir,transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_paths(dir)

    def read_lfw_pairs(self,pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        return pairs
        # return np.array(pairs)

    def get_lfw_paths(self,lfw_dir,file_ext="jpg"):

        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
        #for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
                issame = False

            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list.append((path0,path1,issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1

        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def __getitem__(self, index):
        (path_1, path_2, issame)    = self.validation_images[index]
        image1, image2              = Image.open(path_1), Image.open(path_2)

        # 修复：与训练/部署口径统一，直接 resize 不做 letterbox
        # （LFW 均为 250x250 正方形，两法数值等价，此处仅消除口径歧义）
        image1 = resize_image(image1, [self.image_size[1], self.image_size[0]], letterbox_image = False)
        image2 = resize_image(image2, [self.image_size[1], self.image_size[0]], letterbox_image = False)
        
        image1, image2 = np.transpose(preprocess_input(np.array(image1, np.float32)),[2, 0, 1]), np.transpose(preprocess_input(np.array(image2, np.float32)),[2, 0, 1])

        return image1, image2, issame

    def __len__(self):
        return len(self.validation_images)
