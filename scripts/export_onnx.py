"""
pth -> onnx 导出脚本（导出纯嵌入模型：骨干 + 嵌入头，128 维输出，不含 ArcFace 分类头）

在项目根目录执行：
    python -m scripts.export_onnx
    python -m scripts.export_onnx --model_path <权重.pth> --onnx <输出.onnx>
"""
import os
import argparse
import torch
from nets.facenet import Facenet

# 项目根目录：脚本位于 scripts/ 下，取上一层；权重与输出路径以此为基准
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run():
    print("开始转换模型")
    print(args)

    if not args.model_path.endswith(".pth"):
        print("模型文件格式不正确,model_path=%s" % args.model_path)
        return

    if not os.path.exists(args.model_path):
        print("模型文件不存在,model_path=%s" % args.model_path)
        return

    net = Facenet(backbone="mobilenet", mode="predict").eval()
    net.load_state_dict(torch.load(args.model_path, map_location=args.device), strict=False)
    print('{} model loaded.'.format(args.model_path))

    dummy_input = torch.randn(1, 3, 112, 112, device=args.device)  # 与训练输入一致（112x112）

    # 导出模型到ONNX格式
    input_names = ['input']
    output_names = ['output']

    onnx_dir = os.path.dirname(os.path.abspath(args.onnx))
    os.makedirs(onnx_dir, exist_ok=True)
    if os.path.exists(args.onnx):
        os.remove(args.onnx)
    export_kwargs = dict(
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=10,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    try:
        # torch>=2.9 默认走新导出器（依赖 onnxscript），显式要求经典导出器
        torch.onnx.export(net, dummy_input, args.onnx, dynamo=False, **export_kwargs)
    except TypeError:
        # 旧版 torch（如 2.2）无 dynamo 参数，默认即经典导出器
        torch.onnx.export(net, dummy_input, args.onnx, **export_kwargs)

    if os.path.exists(args.onnx):
        print("模型转换成功,转换文件路径：%s" % args.onnx)
    else:
        print("模型转换失败")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export_onnx')
    parser.add_argument('--model_path', default=os.path.join(PROJECT_ROOT, 'checkpoints', 'model_best_lfw0.9898_ep067.pth'), help='输入模型文件路径，要求pth格式')
    parser.add_argument('--onnx', default=os.path.join(PROJECT_ROOT, 'checkpoints', 'facenet_v4_lfw0.9898.onnx'), help='输出模型文件路径，要求onnx格式')
    parser.add_argument('--device', default="cpu", help='device')

    args = parser.parse_args()
    run()
