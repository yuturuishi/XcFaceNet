# -*- coding: utf-8 -*-
"""
训练历史记录（基于 logging 模块，替代原 TensorBoard/tfevents + matplotlib 方案）

每个 epoch 的损失与精度直接写入本次运行的日志文件（log/train-年月日-时分秒.log），
并在内存中维护历史，用于统计最优精度与最优 epoch。
"""
import logging


class LossHistory():
    def __init__(self):
        self.acc        = []    # 每个 epoch 的评估精度（开启 LFW 时为 LFW 精度，否则为训练集分类精度）
        self.losses     = []    # 每个 epoch 的训练损失
        self.val_loss   = []    # 每个 epoch 的验证损失
        self.best_acc   = -1.0  # 历史最优精度
        self.best_epoch = -1    # 最优精度对应的 epoch（从 1 开始计）

    def append_loss(self, epoch, acc, loss, val_loss):
        self.acc.append(acc)
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        # 更新最优记录
        if acc > self.best_acc:
            self.best_acc   = acc
            self.best_epoch = epoch + 1
            if len(self.acc) > 1:
                logging.info("[历史] 本epoch精度 %.5f 创下新高（此前最优 %.5f）", acc, sorted(self.acc)[-2])

        logging.info(
            "[历史] 累计完成 %d 个epoch | 历史最优精度 %.5f（epoch %03d）",
            len(self.losses), self.best_acc, self.best_epoch,
        )
