# 环境与数据集

本仓库文档与可视化基于 **nuScenes mini**（Occupancy 另需 **Occ3D mini** GT）+ 各算法官方预训练权重推理得到。上游代码与权重不随本仓库分发，需自行准备。

## 1. 硬件与系统建议

| 项 | 建议 |
|---|---|
| GPU | ≥ 24GB 显存（复现环境为 RTX 4090）；DriveLM 7B adapter 推理同量级 |
| 系统 | Linux + NVIDIA 驱动；CUDA 11.7 / 11.8 较常见 |
| 磁盘 | nuScenes mini ≈ 4GB+；若含 sweeps / maps / Occ3D，预留数十 GB |
| Python | 按算法拆 **独立 conda env**，禁止把 mmcv / mmdet3d 版本混装进同一环境 |

## 2. 推荐目录布局

```text
$ROOT/
├── data/
│   ├── nuscenes/              # v1.0-mini
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── maps/
│   │   ├── v1.0-mini/
│   │   └── nuscenes_infos_*.pkl
│   └── occ3d/gts/             # Occ3D mini GT（FB-OCC 评 mIoU）
├── repos/                     # 各上游仓库 clone
│   ├── BEVFormer/
│   ├── StreamPETR/
│   ├── BEVFusion/
│   ├── FB-BEV/
│   ├── VAD/
│   ├── MapTR/
│   ├── Finetune-e2e-model-DriveLM/
│   └── OpenDriveVLA/
├── ckpts/                     # 官方预训练权重
├── results/                   # 推理输出 pkl / json
└── outputs/                   # 可视化、MOT 视频等
```

本仓库（GitHub：`aiwenlan/adas-perception-stack-repro`）可放在 `$ROOT` 旁或单独 clone；目录名建议与仓库名一致。`scripts/` 通过 `--root` / `$AD_ROOT` 指向数据与上游代码工作区。

## 3. 数据集

### 3.1 nuScenes mini（必选）

- 官网：[nuScenes Download](https://www.nuscenes.org/download)
- 版本：`v1.0-mini`（约 10 个 scene，val 常用 **81** frames）
- 解压后目录需能被 `nuscenes-devkit` 识别：`samples` / `sweeps` / `maps` / `v1.0-mini`

生成各仓库所需的 `infos` pkl（名称因项目略有差异），例如：

```bash
# 示例：按各上游 tools/create_data.py 或 create_data.sh 生成
# 常见产物：
#   nuscenes_infos_val.pkl
#   nuscenes_infos_temporal_val.pkl
#   vad_nuscenes_infos_temporal_val.pkl
```

**注意**：本仓库所有表格数字均在 **mini** 上测得，**不能**与论文 full trainval / full val 直接对比。

### 3.2 Occ3D mini（Occupancy 评测）

- 用于 FB-OCC 的 mIoU（本仓库报告 **31.12** @ 81 samples）
- GT 按 token 对齐到 nuScenes sample；放置示例：`$ROOT/data/occ3d/gts/`
- 来源可参考 [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D) / OpenDriveLab 发布渠道

无 Occ3D GT 时仍可做可视化推理，但无法复现 mIoU 数字。

### 3.3 DriveLM / OpenDriveVLA 附加数据

| 用途 | 说明 |
|---|---|
| DriveLM Graph VQA | 需将 nuScenes 图像与 QA 转成 LLaMA-Adapter 所需 JSON（上游 `extract` / `convert2llama` 流程） |
| OpenDriveVLA | 需按官方流程构建 scene / track / map token cache；仅用 mini 时历史轨迹（`his`）可能为空 |

## 4. Conda 环境一览

**原则**：一个算法一个 env；mmcv 带/不带 ops、PyTorch CUDA 版本不要跨 env 复用 site-packages。完整 pin 表见 [docs/08-selection-and-env-matrix.md](docs/08-selection-and-env-matrix.md)。

| Env 名（建议） | 主要用途 | 已知可用组合（复现参考） |
|---|---|---|
| `bevformer` | BEVFormer-small（主） | 按 [BEVFormer](https://github.com/fundamentalvision/BEVFormer) README（常配 mmcv / mmdet3d 1.x） |
| `streampetr` | StreamPETR-R50 | 按 [StreamPETR](https://github.com/exiawsh/StreamPETR)；`flash-attn` 可选，失败可退回普通 attention |
| `bevfusion` | CenterPoint + BEVFusion | 按 [BEVFusion](https://github.com/mit-han-lab/bevfusion)；单机可设 `MASTER_HOST=127.0.0.1:29501` |
| `fbocc` | FB-OCC / FB-BEV | 按 [FB-BEV](https://github.com/NVlabs/FB-BEV) 官方配置 |
| `vad` | VAD-tiny | 按 [VAD](https://github.com/hustvl/VAD) |
| `maptr` | MapTR-tiny | 按 [MapTR](https://github.com/hustvl/MapTR)；`geometric_kernel_attn` 需本地 nvcc 编译 |
| `drivelm` | DriveLM demo | **torch 2.0.0+cu117** 量级；官方 **BIAS-7B** + **LLaMA-1**（勿用 Llama-2） |
| `drivevla` | OpenDriveVLA-0.5B | **torch 2.1.2+cu118** · mmcv 1.7.2 · mmdet3d 1.0.0rc6 · transformers 4.37.2 |
| （轻量） | 本仓库 online MOT | `nuscenes-devkit` · opencv · numpy · pyquaternion；可读检测结果即可，不必装完整检测栈 |

OpenDriveVLA 环境搭建示例思路（细节以官方仓库为准）：

```bash
conda create -n drivevla python=3.10 -y
conda activate drivevla
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# 再按 OpenDriveVLA/third_party 编译或安装 mmcv 1.7.2、mmdet3d 1.0.0rc6
# pip install transformers==4.37.2 ...
```

## 5. 预训练权重

放到 `$ROOT/ckpts/`（或各 repo 的 `pretrained/` / `ckpts/`），名称以官方发布为准：

| 模块 | 权重（示意） |
|---|---|
| BEVFormer-small | `bevformer_small_epoch_24.pth` |
| StreamPETR-R50 | 官方 R50 flash / seq 配置对应 ckpt |
| CenterPoint | BEVFusion 仓库 lidar-only det ckpt |
| BEVFusion | `bevfusion-det.pth`（注意 `add_depth_features` 与 ckpt 一致） |
| FB-OCC | 官方 R50 OCC ckpt |
| VAD-tiny | `VAD_tiny.pth` |
| MapTR-tiny | 官方 tiny ckpt |
| DriveLM | `BIAS-7B.pth` + LLaMA-1 7B consolidated weights |
| OpenDriveVLA | HF：`OpenDriveVLA/OpenDriveVLA-0.5B`（gated，约 1.4GB） |

国内可酌情设置 `HF_ENDPOINT=https://hf-mirror.com` 等镜像。

## 6. 上游仓库

```bash
mkdir -p $ROOT/repos && cd $ROOT/repos
git clone https://github.com/fundamentalvision/BEVFormer.git
git clone https://github.com/exiawsh/StreamPETR.git
git clone https://github.com/mit-han-lab/bevfusion.git BEVFusion
git clone https://github.com/NVlabs/FB-BEV.git
git clone https://github.com/hustvl/VAD.git
git clone https://github.com/hustvl/MapTR.git
git clone https://github.com/aiwenlan/Finetune-e2e-model-DriveLM.git
git clone https://github.com/DriveVLA/OpenDriveVLA.git
```

各仓库内部依赖、编译（CUDA op / deformable attn / geometric kernel）请严格跟上游 README；版本漂一遍是常态。

## 7. 本仓库脚本依赖（在线 MOT）

```bash
pip install nuscenes-devkit opencv-python numpy pyquaternion
```

```bash
# LiDAR-only
python scripts/online_lidar_mot.py \
  --root $ROOT \
  --results-pkl $ROOT/results/centerpoint/results.pkl \
  --scene scene-0103

# LiDAR + Camera 3D
python scripts/online_dual_mot.py \
  --root $ROOT \
  --results-pkl $ROOT/results/centerpoint/results.pkl \
  --streampetr-json $ROOT/results/streampetr/results_nusc.json \
  --scene scene-0103
```

检测结果必须与 `nuscenes_infos_val.pkl` **按 sample_token 对齐**。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| mmcv / mmdet3d ImportError 或 CUDA op 不匹配 | 重装对应 env，勿混用其它 env 的包 |
| BEVFusion DDP 单卡报错 | `export MASTER_HOST=127.0.0.1:29501` |
| MapTR `geometric_kernel_attn` | 需 nvcc 编译；单卡 test 路径需在 config/脚本中打开 |
| DriveLM 乱码 | 确认 backbone 为 **LLaMA-1**，不是 Llama-2 |
| OpenDriveVLA HF 403 | 申请 gated 模型权限；或配置镜像与 token |
| 投影框漂浮 / 陷入地面 | 见 [docs/faq.md](docs/faq.md) 的 z 约定 |
| BEV 框整体横置约 90° | CenterPoint→NuScenes 勿套用 BEVFusion 的 `-yaw-π/2` |

更细的推理命令见 [REPRO.md](REPRO.md)；坐标系与可视化约定见 [docs/faq.md](docs/faq.md)。
