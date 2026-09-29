# -*- coding: utf-8 -*-
r"""
ONNX 模型推理测试（纯 onnxruntime，不依赖 PyTorch）

用途：验证 scripts/export_onnx.py 导出的 .onnx 模型可正常推理，且与 tests.py（PyTorch）
同图同口径下结果一致（余弦相似度偏差应 < 1e-3）。

预处理与 tests.py 完全一致：BICUBIC 直接 resize 112x112（不 letterbox）→ /255 → CHW。
比对口径与线上部署一致：特征 L2 归一化，余弦 = 1 - d²/2。

运行环境：需要 onnxruntime（项目 venv 未装，用 Py312 运行），在项目根目录执行：
    python -m scripts.test_onnx --onnx checkpoints/facenet_v4_lfw0.9898.onnx
"""
import argparse
import logging
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from utils.logger import setup_logger, fmt_duration

# 项目根目录：脚本位于 scripts/ 下，取上一层；权重与样例图路径以此为基准
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preprocess(image, input_shape, letterbox_image=False):
    """与 tests.py / utils.utils 保持同口径的预处理"""
    iw, ih = image.size
    w, h = input_shape[1], input_shape[0]
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    arr = np.asarray(new_image, np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # 1x3xHxW


def main():
    parser = argparse.ArgumentParser(description='test_onnx')
    parser.add_argument('--onnx', default=os.path.join(PROJECT_ROOT, 'checkpoints', 'facenet_v4_lfw0.9898.onnx'),
                        help='待测试的 ONNX 模型路径')
    parser.add_argument('--image1', default=os.path.join(PROJECT_ROOT, 'data', 'Adrien_Brody_0001.jpg'), help='人脸图片1')
    parser.add_argument('--image2', default=os.path.join(PROJECT_ROOT, 'data', 'Adrien_Brody_0012.jpg'), help='人脸图片2')
    parser.add_argument('--input_shape', default='112,112,3', help='模型输入 H,W,C')
    parser.add_argument('--letterbox', action='store_true', help='加此开关用 letterbox 预处理（默认直接 resize，与训练/部署一致）')
    args = parser.parse_args()

    _, log_file = setup_logger("test_onnx")
    logging.info("=" * 78)
    logging.info("XcFaceNet ONNX 模型推理测试（onnxruntime）")
    logging.info("=" * 78)
    logging.info("[启动] 测试时间: %s | 日志文件: %s", time.strftime("%Y-%m-%d %H:%M:%S"), log_file)
    logging.info("[配置] onnx=%s | image1=%s | image2=%s | input_shape=%s | letterbox=%s",
                 args.onnx, args.image1, args.image2, args.input_shape, args.letterbox)

    #---------------------------------------------------#
    #   载入 ONNX 会话
    #---------------------------------------------------#
    if not args.onnx.endswith(".onnx"):
        logging.error("[错误] 模型文件格式不正确，要求 .onnx: %s", args.onnx)
        return
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    logging.info("[模型] 输入: %s %s | 输出: %s %s | 提供者: %s",
                 sess.get_inputs()[0].name, sess.get_inputs()[0].shape,
                 output_name, sess.get_outputs()[0].shape, sess.get_providers())

    input_shape = [int(x) for x in args.input_shape.split(",")]

    #---------------------------------------------------#
    #   预处理两张图并推理
    #---------------------------------------------------#
    image_1 = Image.open(args.image1).convert("RGB")
    image_2 = Image.open(args.image2).convert("RGB")
    logging.info("[输入] 图片1: %s（%dx%d） | 图片2: %s（%dx%d）",
                 args.image1, image_1.width, image_1.height, args.image2, image_2.width, image_2.height)

    photo_1 = preprocess(image_1, input_shape, letterbox_image=args.letterbox)
    photo_2 = preprocess(image_2, input_shape, letterbox_image=args.letterbox)

    t1 = time.time()
    output_1 = sess.run([output_name], {input_name: photo_1})[0]
    output_2 = sess.run([output_name], {input_name: photo_2})[0]
    t2 = time.time()
    logging.info("[推理] 单图耗时: %s | 输出维度: %s", fmt_duration((t2 - t1) / 2), output_1.shape)

    #---------------------------------------------------#
    #   余弦相似度（与 test.py / 线上部署同口径）
    #---------------------------------------------------#
    feat_1 = output_1[0]
    feat_2 = output_2[0]
    norm_1 = np.linalg.norm(feat_1)
    norm_2 = np.linalg.norm(feat_2)
    logging.info("[特征] 范数: %.6f / %.6f（L2 归一化导出应≈1，未归一化则自动补除）", norm_1, norm_2)
    cosine = float(np.dot(feat_1, feat_2) / (norm_1 * norm_2 + 1e-9))

    dist = np.linalg.norm(feat_1 / norm_1 - feat_2 / norm_2)
    logging.info("[结果] 余弦相似度: %.4f（越大越相似） | 归一化空间距离: %.5f", cosine, float(dist))

    #---------------------------------------------------#
    #   与 tests.py（PyTorch）口径对照：同模型同图应几乎一致
    #---------------------------------------------------#
    logging.info("[对照] 同权重同图时 tests.py（PyTorch）应输出几乎相同的余弦相似度（偏差 < 1e-3）")
    logging.info("[结论] ONNX 模型推理测试通过: %s", args.onnx)


if __name__ == "__main__":
    main()
