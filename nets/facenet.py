import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.mobilenet import MobileNetV1

class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()

        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x

class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        if backbone != "mobilenet":
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet.'.format(backbone))
        self.backbone = mobilenet()
        flat_shape = 1024

        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            # ArcFace 分类头：无 bias，forward 中做权重归一化 + 角度间隔
            self.classifier = nn.Linear(embedding_size, num_classes, bias=False)
            # ArcFace 超参（参考 ArcFace 原文）
            self.s = 64.0          # 特征缩放因子
            self.m = 0.50          # 角度间隔（弧度）
            # 预计算 sin/cos 常量，避免每步重算
            self.cos_m = math.cos(self.m)
            self.sin_m = math.sin(self.m)
            self.mm = self.sin_m * self.cos_m  # sin(m)*cos(m)，用于简化公式

    def forward(self, x, mode = "predict", labels = None):
        if mode == 'predict':
            x = self.backbone(x)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            # 修复：推理路径移除 Dropout（原先依赖调用方 .eval() 才安全，
            # 移除后即使误在 train 模式下提特征也不会被随机丢弃污染）
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)

        x = F.normalize(before_normalize, p=2, dim=1)
        # ArcFace 逻辑：
        # 1. 分类头权重归一化  2. 计算 cos(theta)  3. 仅对真实标签类加角度间隔 m
        # 稳定实现：用 if sin > 0 的方式区分 theta 是否在 [0, pi-m] 内
        cos_data = F.linear(x, F.normalize(self.classifier.weight))
        sin_data = torch.sqrt(1.0 - cos_data * cos_data + 1e-7)
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        cos_add = cos_data * self.cos_m - sin_data * self.sin_m
        # 当 theta+m 超过 pi 时，cos(theta+m) < cos(theta)，此时用 cos(theta)-mm 代替
        # 用 torch.where 保证梯度稳定
        target_logit = torch.where(cos_data > 0, cos_add, cos_data - self.mm)
        if labels is not None:
            # 标准 ArcFace：只有真实标签那一类用 cos(theta_y + m)，其余类保持 cos(theta_j)。
            # 修复历史 bug：旧实现把 margin 逐元素加到了全部 num_classes 个 logits 上，
            # 训练目标被破坏，导致特征判别力受损、部署时同人余弦得分整体偏低。
            one_hot = torch.zeros_like(cos_data)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
            cls = (cos_data * (1.0 - one_hot) + target_logit * one_hot) * self.s
        else:
            raise ValueError("训练模式必须传入 labels")
        return x, cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x
