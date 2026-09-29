# -*- coding: utf-8 -*-
"""
统一日志系统（基于 Python logging 标准库，替代原 TensorBoard/tfevents 方案）

- 日志文件统一存放于项目根目录 log/ 文件夹（不存在则自动创建）
- 命名规则：<mode>-年月日-时分秒.log，如 train-20260913-153045.log、test_pytorch-20260916-161043.log
- mode 为日志前缀，只允许小写字母/下划线（train / test_pytorch / test_onnx 等）
- 每次运行生成一个独立日志文件，同时输出到控制台与文件
- 默认输出所有级别（DEBUG 及以上）的日志
"""
import logging
import os
import re
import sys
from datetime import datetime

# 项目根目录：log/ 等路径以项目根为基准，在任意目录执行都不会跑偏
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logger(mode="train", log_dir=None):
    """
    初始化全局日志系统（配置 root logger，之后全项目可直接使用 logging.info 等调用）

    :param mode:    日志类型：'train'（训练）或 'test'（测试）
    :param log_dir: 日志根目录，默认取项目根目录下的 log 文件夹（绝对路径）
    :return:        (logger, 日志文件绝对路径)
    """
    if not re.match(r"^[a-z][a-z_]*$", mode):
        raise ValueError("mode 只允许小写字母/下划线（用作日志文件名前缀），当前传入: %s" % mode)

    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, "log")

    # Windows 控制台默认 GBK 编码，重定向为 UTF-8 防止中文日志乱码/报错
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file  = os.path.abspath(os.path.join(log_dir, "%s-%s.log" % (mode, timestamp)))

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)   # 输出所有级别
    logger.handlers.clear()          # 防止重复初始化时叠加 handler

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件输出：所有级别
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台输出：所有级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_file


def fmt_duration(seconds):
    """把秒数格式化为 'x小时x分x秒' 形式，便于日志阅读"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return "%d小时%d分%d秒" % (h, m, s)
    if m > 0:
        return "%d分%d秒" % (m, s)
    return "%d秒" % s
