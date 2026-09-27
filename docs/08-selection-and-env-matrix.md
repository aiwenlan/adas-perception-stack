# 08 · 算法选型 · 数据选型 · 云端配置 · Conda 版本矩阵

> 本仓库复现实验的**拍板清单**。选型一旦固定，不临时换主栈；环境**一算法一 conda**，禁止混装 mmcv / torch。  
> 与执行计划「〇章」一致，并补上后续实跑的 OpenDriveVLA / 在线 MOT。

## 1. 算法选型

### 1.1 主选总表

| 任务 | **主选（必须跑通）** | **对照（只读 / 不强制跑）** | 选型理由 |
|---|---|---|---|
| Camera BEV 检测（Dense Query） | **BEVFormer-small** | tiny（轻量对照）；base（官方训练显存 ~28.5GB，4090 不做主跑）；LSS/BEVDet | Query-BEV 正统；small 精度与 24GB 平衡好 |
| Camera 3D（Sparse + 时序） | **StreamPETR-R50** | PETR / DETR3D / Sparse4D；V2-99 仅不 OOM 时试 | 开源 camera temporal 标杆；与 dense BEV 对照 |
| LiDAR 3D | **CenterPoint** | PointPillars / SECOND | 工业界点云检测事实标准 |
| Cam+LiDAR 融合检测 | **BEVFusion**（batch=1） | TransFusion / FUTR3D | BEV 空间融合经典；本仓库 **跟踪主路径不用它** |
| Occupancy | **FB-OCC R50** | SurroundOcc / SparseOcc | NVIDIA 开源；有 TRT 部署路径；Occ3D 指标齐 |
| 端到端规划 | **VAD-tiny** | UniAD（只读架构） | 更轻，适配 24GB；UniAD 作系统教材 |
| 矢量地图 | **MapTR-tiny** | VectorMapNet / LaneSegNet | BEV→矢量地图面试高频 |
| VLM | **DriveLM**（自有微调仓 + BIAS-7B 推理） | OmniDrive | Graph VQA；LLaMA-Adapter |
| VLA | **OpenDriveVLA-0.5B**（推理展示） | OccVLA 等大权重（默认可只读） | 24GB 可演示轨迹 grounding |
| 在线 3D MOT | **KF + 门控 NN**（CenterPoint ± StreamPETR） | 学习型 TrackFormer 等 | 模块化可讲；E2E live 已测延迟 |

**主栈一句话**  
`CenterPoint → BEVFormer / StreamPETR →（可选 BEVFusion）→ FB-OCC → VAD + MapTR → DriveLM / OpenDriveVLA → DualTracker MOT`

### 1.2 4090 上跑 / 不跑边界

| 等级 | 内容 |
|---|---|
| **必须跑通** | BEVFormer-small、StreamPETR-R50、CenterPoint、BEVFusion(batch=1)、FB-OCC-R50、VAD-tiny、MapTR、DriveLM `demo.py`、在线 MOT（含 E2E live） |
| **已补充实跑** | OpenDriveVLA-0.5B mini 推理可视化 |
| **默认不跑** | UniAD 全量、StreamPETR-ViT-L、≥7B 新 VLA 全精度、BEVFormer-base 主训 |

### 1.3 24GB 防 OOM

1. 一律 `batch_size=1`。  
2. 优先 fp16；推理与可视化分开，跑完 `empty_cache()`。  
3. 同时只跑一个重仓库。  
4. 分辨率 / 时序用官方 tiny·small / 最短可复现。  
5. 连续 OOM 两次 → 降配或改只读，**不换更大 GPU、不换未拍板模型**。

---

## 2. 数据选型

| 用途 | 数据集 | 说明 |
|---|---|---|
| **主数据** | **nuScenes mini**（本仓库全部表格） | 6 相机 + LiDAR + 标定；val 常用 **81** frames |
| 全量训练/对标论文 | nuScenes trainval（按需） | 本仓库**不对标** full-val 论文表 |
| Occupancy GT | **Occ3D-nuScenes mini** | FB-OCC mIoU；`gts/<scene>/<token>/labels.npz` |
| E2E / Map | nuScenes + 各仓 temporal infos / map ann | VAD、MapTR converter |
| VLM | DriveLM-nuScenes + 自有 convert2llama | Graph VQA JSON |
| VLA | OpenDriveVLA 官方 cache 流程 | mini 上 `his` 可能为空 |
| 对照（可不下） | KITTI / Waymo 指标差异 | 面试口述即可 |
| 闭环（只读） | NAVSIM / Bench2Drive / CARLA | open-loop vs closed-loop |

**磁盘预算（云盘）**

| 项 | 量级 |
|---|---|
| nuScenes mini + 预处理 | ~40–80 GB |
| 补 val 图 + sweeps + Occ3D | ~300–500 GB（按需） |
| 各模型权重 | ~20–40 GB |
| **建议预留** | **≥ 800 GB NVMe**（本复现机按此租） |

数据如何采集 / infos 如何建：见 [06 · 数据与训练](06-data-and-training.md)。

---

## 3. 云端服务器配置（唯一方案）

| 项 | 规格 |
|---|---|
| **GPU** | **1 × NVIDIA RTX 4090 24GB** |
| **OS** | **Ubuntu 22.04 LTS**（备选 20.04） |
| **Driver** | **≥ 535**（推荐 535 / 550） |
| **系统 CUDA Toolkit（nvcc）** | **11.8（唯一；禁止 12.x）** |
| **CPU** | ≥ 12–16 vCPU |
| **RAM** | ≥ **64 GB**（推荐 96–128 GB） |
| **系统盘** | 50–100 GB |
| **数据盘** | **≥ 800 GB NVMe**（推荐 1 TB） |
| **租用** | 连续 48–60 h 量级 |
| **平台** | AutoDL / 恒源云 / Vast.ai / RunPod 等单卡 4090 |

### 3.1 为何锁死 CUDA 11.8

| 因素 | 结论 |
|---|---|
| 4090 / sm_89 | Driver 要新；Toolkit 11.8 友好 |
| 感知老栈 | BEVFormer / VAD / MapTR / StreamPETR / BEVFusion 依赖 **mmcv-full** 预编译轮；**CUDA 12 镜像坑最多** |
| DriveLM | `torch==2.0.0+cu117`，与系统 11.x 一族兼容 |
| 原则 | **机器只装一套 nvcc 11.8**；各 env 自带 `cu111/cu113/cu117` wheel，**不追求一个 torch 打天下** |

### 3.2 租机镜像怎么选

```text
优先：Ubuntu 22.04 + CUDA 11.8 + 任意基础 PyTorch（进机后不用镜像自带 torch）
禁止：CUDA 12.1 / 12.2 / 12.4 镜像

进机验收：
  nvidia-smi    # 4090 + Driver≥535
  nvcc -V       # release 11.8
  df -h         # 数据盘 ≥800GB
```

### 3.3 目录布局（与 SETUP 一致）

```text
$ROOT/
├── data/nuscenes/     # mini
├── data/occ3d/gts/
├── repos/             # 各上游 clone
├── ckpts/
├── results/
├── outputs/
└── adas-perception-stack-repro/  # 本仓库（文档、evidence、MOT 脚本）
```

---

## 4. Conda 环境版本矩阵

**铁律**：一算法一 env；禁止把 mmcv / mmdet3d 跨 env 复用。

| conda env | 算法 | Python | PyTorch | mmcv / mmdet 关键 pin | 备注 |
|---|---|---|---|---|---|
| `bevformer` | BEVFormer-small（主） | **3.8** | **1.9.1+cu111** | mmcv-full==1.4.0；mmdet==2.14.0；mmseg==0.14.1；mmdet3d **v0.17.1** | 官方 install；batch=1 |
| `streampetr` | StreamPETR-R50；E2E MOT 同进程加载 CP+SP | **3.8** | **1.9.0+cu111** | mmcv-full==1.6.0；mmdet==2.28.2；mmseg==0.30.0；mmdet3d **v1.0.0rc6** | flash-attn 可选，失败改普通 attn |
| `bevfusion` | CenterPoint + BEVFusion | **3.8** | **1.10.2+cu113** | mmcv==1.4.0；mmdet==2.20.0 | `setup.py develop` 用系统 nvcc 11.8；`MASTER_HOST=127.0.0.1:29501` |
| `fbocc` | FB-OCC | **3.8** | **1.12.0+cu113** | mmcv-full==1.5.2；mmdet==2.24.0；mmseg==0.24.0 | Occ3D `occupancy_path` |
| `vad` | VAD-tiny | **3.8** | **1.9.1+cu111** | 同 BEVFormer 族：mmcv-full==1.4.0；mmdet3d v0.17.1 | 仍建议独立 env |
| `maptr` | MapTR | **3.8** | **1.9.1+cu111** | mmcv-full==1.4.0；mmdet==2.14.0；mmseg==0.14.1 | 编译 geometric_kernel_attn |
| `drivelm` | DriveLM | **3.8** | **2.0.0+cu117** | 无 mmcv；按仓库 requirements | **LLaMA-1** backbone；batch_size=1 |
| `drivevla` | OpenDriveVLA-0.5B | **3.10** | **2.1.2+cu118** | mmcv 1.7.2；mmdet3d 1.0.0rc6；transformers 4.37.2 | HF gated 权重 |
| （轻量） | online MOT 回放 | 3.8+ | 任意 CPU/GPU | 无 mmcv | nuscenes-devkit、opencv、numpy、pyquaternion |

### 4.1 速记：谁用哪张 torch

```text
bevformer / vad / maptr  →  torch 1.9.1+cu111
streampetr               →  torch 1.9.0+cu111
bevfusion(+CenterPoint)  →  torch 1.10.2+cu113
fbocc                    →  torch 1.12.0+cu113
drivelm                  →  torch 2.0.0+cu117
drivevla                 →  torch 2.1.2+cu118
```

### 4.2 4090 + 老 PyTorch 1.9 排障

若 sm_89 / kernel 报错，按序：

1. `export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"`  
2. 仅该 env 升到 **1.10.2+cu113**，并同步 mmcv 轮子  
3. 仍失败记 blocker  

**禁止**为省事把系统升到 CUDA 12 / 全机 torch 2.1。

### 4.3 创建 env 示例（OpenDriveVLA）

```bash
conda create -n drivevla python=3.10 -y
conda activate drivevla
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# 再按 OpenDriveVLA 安装 mmcv 1.7.2、mmdet3d 1.0.0rc6、transformers==4.37.2
```

其余 env 严格跟各上游 README + 上表 pin；wheel 源优先官方 / OpenMMLab 对应 cu 版本。

---

## 5. 上游仓库（clone 清单）

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

---

## 6. 与其它文档

| 文档 | 关系 |
|---|---|
| [SETUP.md](../SETUP.md) | 安装步骤与目录 |
| [REPRO.md](../REPRO.md) | 推理命令 |
| [06 · 数据与训练](06-data-and-training.md) | 采集与建库细节 |
| [07 · 量产部署加速](07-deployment-acceleration.md) | 车端 TRT，非云端租机 |
| [05 · 技术栈总览](05-stack-overview.md) | 指标与链路 |

**一句话**：算法与数据按上表拍板；云端 **1×4090 + Ubuntu22.04 + CUDA11.8 + ≥800GB**；每个算法独立 conda，按矩阵 pin torch/mmcv，才能稳定复现本仓库指标与可视化。
