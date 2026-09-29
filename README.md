# XcFaceNet

> 轻量级人脸特征提取与比对工程：MobileNetV1 + 128 维嵌入，TripletLoss + ArcFace 联合监督，训练 / 测试 / 导出 / 部署全链路口径一致。

| | |
|---|---|
| 官网 | https://www.yuturuishi.com |
| 微信 | yuturuishi |
| Gitee | https://gitee.com/yuturuishi/XcFaceNet |
| GitHub | https://github.com/yuturuishi/XcFaceNet |

---

## 目录结构

```
XcFaceNet/
├── train.py                     训练入口（checkpoint 目录自动递增 + --resume 断点续训）
├── tests.py                     测试入口（PyTorch 口径，两图余弦相似度）
├── requirements.txt             依赖清单（Windows / Linux 通用）
├── nets/                        网络定义
│   ├── mobilenet.py             MobileNetV1 骨干（深度可分离卷积）
│   ├── facenet.py               Facenet：骨干 + 1024→128 嵌入头 + ArcFace 分类头
│   └── facenet_training.py      TripletLoss / 学习率调度 / 权重初始化
├── utils/                       库代码
│   ├── dataloader.py            FacenetDataset（在线 triplet 采样）、LFWDataset
│   ├── utils_fit.py             单 epoch 训练 / 验证 / LFW 评估 / checkpoint 保存
│   ├── utils_metrics.py         LFW 10 折交叉验证
│   ├── logger.py                统一日志（log/ 下按时间戳命名）
│   ├── callback.py              训练历史与最优精度记录
│   └── utils.py                 预处理 / 随机种子 / 配置打印
├── scripts/                     工具脚本，统一 python -m scripts.<脚本名> 运行
│   ├── export_onnx.py           .pth → .onnx（纯嵌入模型，不含分类头）
│   ├── test_onnx.py             ONNX 推理测试（纯 onnxruntime，不依赖 PyTorch）
│   ├── create_train.py          由图片目录生成训练描述文件 train.txt
│   ├── create_lfw.py            由 LFW 图片目录生成 6000 对 lfw.txt
│   └── unpack_faces_emore.py    faces_emore(.rec/.idx) 解包为图片目录 + train.txt
├── data/                        测试样例图片
├── checkpoints/                 训练产物与导出模型（本地生成，不纳入版本管理）
├── log/                         训练与测试日志（本地生成，不纳入版本管理）
├── venv/                        虚拟环境（不纳入版本管理）
├── LICENSE / .gitignore
└── README.md
```

根目录只保留两个入口：`train.py`（训练）、`tests.py`（测试），其余脚本统一放在 `scripts/`。

> `.gitignore` 已屏蔽全部模型文件与产物（`checkpoints*/`、`log/`、`datasets/`、`venv/`、`__pycache__/`，
> 以及 `*.pth`、`*.pt`、`*.ckpt`、`*.onnx`、`*.engine`、`*.plan`、`*.trt`、`*.h5`、`*.safetensors` 等），
> 模型文件不会进入版本库。

---

## 环境安装

* 建议 Python 3.10 ~ 3.12（当前环境 Python 3.12.3 + CUDA 12.1，`requirements.txt` 按该环境冻结）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate

python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* `requirements.txt` 内置 PyTorch 的 cu121 构建源；需其他 CUDA 版本时单独换装：
  `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
* 纯 CPU 也可训练与评测（自动降级，只是慢）；训练与 LFW 评估在 GPU 上快得多
* ONNX 导出与 ONNX 测试需额外的 `onnx onnxruntime`，建议放在独立环境执行：
  新版 onnx 会强制升级 numpy 到 2.x，与 venv 内按 numpy 1.x 编译的 torch 2.2 冲突，故不装进 venv

---

## 数据准备

| 用途 | 数据集 | 规模 | 磁盘 | 说明 |
|---|---|---|---|---|
| 训练 | MS1MV2 | 85,742 身份 / 5,822,657 张 | ~14 GB | 112×112 RGB，已五点对齐，无需再裁剪 |
| 评估 | LFW | 4,426 身份 / 10,263 张 / 6,000 对 | 23.8 MB | 官方 LFW 的验证对子集，仅评估，不参与训练 |

### 数据集来源

**MS1MV2（训练集）** —— 由微软 MS-Celeb-1M 经 InsightFace 清洗 + RetinaFace 五点对齐得到（即 ArcFace 论文所用的 MS1M-ArcFace，内部名 `ms1m-retinaface-t1`），统一对齐到 112×112；官方以 `faces_emore` 压缩包发布，内部为 MXNet RecordIO 格式（`train.idx` / `train.rec`）。

- 下载：[InsightFace 官方数据集页](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) → `MS1M-ArcFace (85K ids/5.8M images)`，提供百度网盘 / Google Drive 两种链接

**LFW（评估集）** —— Labeled Faces in the Wild（UMass Amherst, 2007），新闻场景下的无约束人脸验证基准；官方完整版为 **13,233 张 / 5,749 身份**（250×250 原图）。

- 下载：http://vis-www.cs.umass.edu/lfw/lfw.tgz
- **本项目使用的是与其 6,000 对验证协议相关的子集（4,426 身份 / 10,263 张）**，故直接下载官方全量时规模数字对不上，属正常

### 转换为可训练格式

MS1MV2 是 MXNet `.rec/.idx`，需先解包成「图片目录 + 描述文件」：

```bash
# faces_emore → 图片目录 + train.txt（约 580 万张，建议后台运行，中断后重跑同一命令即可续传）
python -m scripts.unpack_faces_emore --src <faces_emore目录> --dst <图片输出目录> --train_txt <train.txt路径> --all
python -m scripts.create_train     # 已有图片目录时，仅重新生成 train.txt
python -m scripts.create_lfw       # LFW 图片目录 → lfw.txt（10 折 × 300 对，随机配对）
```

* 描述文件格式：`train.txt` 每行 `label;图片绝对路径`；`lfw.txt` 首行 `10 300`，随后 6,000 行，同人对 `人名 图号1 图号2`、异人对 `人名1 图号1 人名2 图号2`
* 三个脚本的路径写在文件顶部或参数默认值中，数据换位置需同步修改；`create_train.py` / `create_lfw.py` 会覆盖已有描述文件
* 路径改完，`train.py` 顶部三处配置需同步：`train_desc_path`、`lfw_path`、`lfw_desc_path`

---

## 模型训练

配置集中在 `train.py` 顶部（数据路径、epoch、学习率、batch_size 等），直接改文件再启动；默认**从头训练**（随机初始化，不加载任何预训练权重）。

```bash
python train.py             # 全新训练（输出到自增新目录）
python train.py --resume    # 迭代训练（从最新 checkpoint 继续，续写原目录）
```

* **checkpoint 目录自动递增**：根目录为 `checkpoints`，每次启动若已存在则依次使用 `checkpoints1`、`checkpoints2`……（取第一个空缺号），每次训练独立目录，**历史模型永不覆盖**
* `--resume` 自动搜索 `checkpoints*` 全部目录里修改时间最新的 checkpoint，从该 epoch 继续并续写原目录；不加参数 = 全新训练
* checkpoint **只保存 `.pth`**（state_dict，约 55MB/个）：训练 / 断点续训 / 测试 / 导出 ONNX 全链路统一只用 `.pth`
* 每 epoch 保存（文件名含损失与 LFW 精度），LFW 新高自动另存 `model_best_*.pth`

GPU 吞吐配置（RTX 3080 Ti 16GB 实测）：`batch_size=384`（128 个 triplet/步，须为 3 的倍数）、每 epoch 在线采样 60 万 triplet、fp16 混合精度、`num_workers=6` + 持久化预取——稳态 0.38 秒/步、峰值显存约 3.7GB。

训练产物：

```
checkpoints/
├── model_epoch067_train_loss17.9865_val_loss13.9267_lfw_acc0.9898.pth   # 逐 epoch 权重
└── model_best_lfw0.9898_ep067.pth                                       # LFW 最优权重
```

### 导出推理模型

导出**纯嵌入模型**：只保留骨干 + 嵌入头（128 维归一化输出），**不含训练专用的 ArcFace 分类头**，输入 1×3×112×112，动态 batch。

```bash
# 默认导出最优权重 → checkpoints/facenet_v4_lfw0.9898.onnx（约 12.7MB）
python -m scripts.export_onnx
# 指定权重与输出路径
python -m scripts.export_onnx --model_path checkpoints/model_best_lfw0.9898_ep067.pth --onnx checkpoints/facenet_v4_lfw0.9898.onnx
```

---

## 推理测试

两个口径各一个脚本，**预处理与比对公式完全一致**，用于互相验证（同权重同图余弦偏差 < 1e-3）：

| 脚本 | 运行环境 | 依赖 |
|---|---|---|
| `tests.py` | 项目 venv | PyTorch（加载 .pth） |
| `python -m scripts.test_onnx` | 任意装了 onnxruntime 的 Python | onnxruntime（纯推理，不依赖 PyTorch） |

```bash
# PyTorch 口径（项目 venv）
python tests.py
# ONNX 口径（项目 venv 未装 onnxruntime，用独立环境运行）
python -m scripts.test_onnx --onnx checkpoints/facenet_v4_lfw0.9898.onnx   # 需先按上节导出
```

* 默认测试 `data/` 下两张样例图，输出余弦相似度（越大越相似，与线上部署同口径）
* 换模型：`tests.py` 改 `__params["model_path"]`；`scripts/test_onnx.py` 用 `--onnx` 参数
* 换图片：`tests.py` 改 `url1 / url2`；`scripts/test_onnx.py` 用 `--image1 / --image2`
* 实测日志见 `log/test_pytorch-*.log`、`log/test_onnx-*.log`；余弦值只有在**同权重同图片**下才具备口径可比性

---

## 部署对齐口径

* 输入：人脸检测 → 五点关键点对齐（`estimateAffinePartial2D` 到 ArcFace 标准参考点）→ 112×112，**不做 letterbox**，直接 resize（与训练一致）
* 推理：可加水平翻转 TTA（翻转特征与原特征取均值）后再 L2 归一化
* 比对：余弦相似度，**经验阈值 0.40**（同人 0.563~0.897 / 异人 max 0.129，间隔 +0.434）
* 匹配度百分比展示用 sigmoid 校准：`100 / (1 + exp((0.35 - cos) / 0.085))`，前后端同公式

---

## 训练结果

MS1MV2 全量 582 万张，100 epoch 完整跑完：

| 指标 | 数值 |
|---|---|
| 最优 LFW 准确率 | **0.98983** → `checkpoints/model_best_lfw0.9898_ep067.pth`（epoch 054 / 067 均达 0.9898） |
| 最终 epoch 100 | LFW 0.98833 \| train_loss 17.7126 \| val_loss 13.6311 \| val_acc 11.09% |
| 每 epoch 规模 | 4,687 步 × 384 图 = 1,799,808 图（60 万 triplet 在线采样） |
| 每 epoch 耗时 | 约 31~34 分钟（训练约 29 分 + 验证约 1.5 分 + LFW 评估 12~18 秒） |
| 全流程耗时 | 98 个 epoch 共 54 小时 57 分（稳态 0.38 秒/步） |
| 收敛情况 | 最优出现在中段（epoch 54~67），之后 30+ epoch 未再提升，已收敛 |

> `val_acc` 是 ArcFace 分类头在 85,742 类上的 top-1 准确率（每 epoch 随机采样、非全量），数值本就只有约 11%，**不代表模型质量**；判断人脸判别力请以 **LFW 准确率**为准。

---

## 预处理说明

| 环节 | 处理 |
|---|---|
| 输入 | 检测 + 五点对齐后的 112×112 RGB 人脸图，直接 resize（不 letterbox） |
| 归一化 | `img / 255`（不做 mean/std 标准化） |
| 输出 | 128 维嵌入，L2 归一化到单位超球面；余弦 = `1 - d²/2` |

---

## 开源协议

本项目以 **MIT 协议** 开源，详见仓库根目录 `LICENSE` 文件（Copyright 2025 yuturuishi）。
