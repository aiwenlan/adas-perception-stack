# 复现指南

本文把本仓库结果分成三类：

- **Evaluation**：官方预训练权重在 nuScenes mini / Occ3D mini 上运行官方评测入口。
- **Visualization**：读取 evaluation 产物，按 `sample_token` 对齐后绘图。
- **Demo**：只证明推理链路可执行，不等价于论文 full-val 指标或闭环效果。

环境、数据、权重和目录布局见 [SETUP.md](SETUP.md)。下文假设：

```bash
export ROOT=/path/to/autonomous_driving_upgrade
export CUDA_VISIBLE_DEVICES=0
```

所有命令均为单卡、`batch_size=1`。上游仓库和权重不随本仓库分发；不同上游 commit 的配置文件名可能变化，因此复现时应同时记录 `git rev-parse HEAD`。

## 0. 每次运行先保存证据

```bash
mkdir -p "$ROOT/outputs/repro"
nvidia-smi | tee "$ROOT/outputs/repro/nvidia-smi.txt"
nvcc -V | tee "$ROOT/outputs/repro/nvcc.txt"
cat /etc/os-release | tee "$ROOT/outputs/repro/os-release.txt"
python -V | tee "$ROOT/outputs/repro/python.txt"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" \
  | tee "$ROOT/outputs/repro/torch.txt"
```

对每个上游仓库再保存：

```bash
git rev-parse HEAD
conda env export --from-history
sha256sum /path/to/checkpoint.pth
```

仓库表格中的数字只有在同时保留 `command.txt`、`eval.log`、config、checkpoint SHA256 和 upstream commit 时，才算完整证据。

## 1. BEVFormer-small

```bash
conda activate bevformer
cd "$ROOT/repos/BEVFormer"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6+PTX}"

mkdir -p "$ROOT/outputs/bevformer-small"
python tools/test.py \
  projects/configs/bevformer/bevformer_small.py \
  "$ROOT/ckpts/bevformer/bevformer_small_epoch_24.pth" \
  --eval bbox \
  2>&1 | tee "$ROOT/outputs/bevformer-small/eval.log"
```

本仓库记录的 mini-val 结果为 mAP **0.3541**、NDS **0.3989**。tiny 的 0.2649/0.3255 只作为轻量对照，不能用 tiny 命令冒充 small 结果。

## 2. StreamPETR-R50

成功路径使用 `stream_petr_r50_flash_704_bs2_seq_90e.py`，单卡时把 config 的 batch 改成 1；无法安装 `flash-attn` 时将 `PETRMultiheadFlashAttention` 切到普通 `PETRMultiheadAttention`。

```bash
conda activate streampetr
cd "$ROOT/repos/StreamPETR"
export PYTHONPATH="$PWD:$PWD/mmdetection3d:${PYTHONPATH:-}"

mkdir -p "$ROOT/outputs/streampetr-r50"
python tools/test.py \
  projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
  ckpts/stream_petr_r50_flash_704_bs2_seq_90e.pth \
  --eval bbox \
  --out "$ROOT/outputs/streampetr-r50/results.pkl" \
  2>&1 | tee "$ROOT/outputs/streampetr-r50/eval.log"
```

部分 upstream 版本的 `tools/test.py` 主动禁用了 non-distributed 分支；若出现 `assert False`，需按该 commit 的 distributed test 命令运行，或仅恢复官方 `MMDataParallel + single_gpu_test` 分支。任何本地 patch 都应通过 `git diff > upstream.patch` 保存，不能只写“改一下 test.py”。

## 3. CenterPoint

本仓库的 CenterPoint 数字来自 OpenMMLab voxel 0.1 配置，不是 BEVFusion 表格中的 lidar-only 模型。

```bash
conda activate streampetr
cd "$ROOT/repos/StreamPETR/mmdetection3d"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "$ROOT/outputs/centerpoint"
python tools/test.py \
  configs/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py \
  "$ROOT/ckpts/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.pth" \
  --eval bbox \
  --out "$ROOT/outputs/centerpoint/results.pkl" \
  2>&1 | tee "$ROOT/outputs/centerpoint/eval.log"
```

## 4. BEVFusion lidar-only 与 camera+lidar

```bash
conda activate bevfusion
cd "$ROOT/repos/BEVFusion"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MASTER_HOST=127.0.0.1:29501

mkdir -p "$ROOT/outputs/bevfusion"

# lidar-only 对照
python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  pretrained/lidar-only-det.pth \
  --eval bbox \
  2>&1 | tee "$ROOT/outputs/bevfusion/eval-lidar.log"

# camera + lidar
python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox \
  --out "$ROOT/outputs/bevfusion/results.pkl" \
  2>&1 | tee "$ROOT/outputs/bevfusion/eval-fusion.log"
```

`add_depth_features` 必须与 checkpoint 对应配置一致。若上游要求 `torchpack dist-run -np 1`，用它包裹同一条 `tools/test.py` 命令。

## 5. FB-OCC / Occ3D

```bash
conda activate fbocc
cd "$ROOT/repos/FB-BEV"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CFG=occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.py
CKPT="$ROOT/ckpts/fbocc/fbocc-r50-cbgs_depth_16f_16x4_20e.pth"
OUT="$ROOT/outputs/fbocc"
mkdir -p "$OUT"

python tools/test.py "$CFG" "$CKPT" \
  --eval occupancy \
  --out "$OUT/results_miou.pkl" \
  --launcher none \
  2>&1 | tee "$OUT/eval.log"
```

评 mIoU 前必须确认 config 的 `occupancy_path` 指向 `$ROOT/data/occ3d/gts`，并记录 `labels.npz` 数量。本仓库报告的 **31.12** 是 81 个 mini-val sample、`mask_camera` 口径；不是官方 full-val 数字。可视化的去噪和 Z 向投票不参与指标计算。

## 6. VAD-tiny

```bash
conda activate vad
cd "$ROOT/repos/VAD"
export PYTHONPATH="$PWD:$ROOT/repos/mmdetection3d_vad:${PYTHONPATH:-}"

OUT="$ROOT/outputs/vad"
mkdir -p "$OUT"
python tools/test.py \
  projects/configs/VAD/VAD_tiny_stage_2.py \
  ckpts/VAD_tiny.pth \
  --launcher none \
  --eval bbox \
  --tmpdir "$OUT/tmp" \
  --out "$OUT/results.pkl" \
  2>&1 | tee "$OUT/eval.log"
```

VAD 依赖 nuScenes map expansion 和 temporal infos。报告 planning L2 时必须同时注明 horizon、sample 数和 open-loop 限制；`collision=0` 在 mini 子集上不代表闭环安全。

## 7. MapTR-tiny

```bash
conda activate maptr
cd "$ROOT/repos/MapTR"
export PYTHONPATH="$PWD:$PWD/mmdetection3d:${PYTHONPATH:-}"

cd projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
cd "$ROOT/repos/MapTR"

OUT="$ROOT/outputs/maptr"
mkdir -p "$OUT"
python tools/test.py \
  projects/configs/maptr/maptr_tiny_r50_24e.py \
  ckpts/maptr_tiny_r50_24e.pth \
  --launcher none \
  --eval chamfer \
  --tmpdir "$OUT/tmp" \
  --out "$OUT/results.pkl" \
  2>&1 | tee "$OUT/eval.log"
```

和 StreamPETR 类似，部分 upstream commit 禁用了单卡 test 分支；应保存对应 patch 或使用该 commit 推荐的 launcher。

## 8. DriveLM

```bash
conda activate drivelm
cd "$ROOT/repos/Finetune-e2e-model-DriveLM/challenge/llama_adapter_v2_multimodal7b"

OUT="$ROOT/outputs/drivelm"
mkdir -p "$OUT"
python demo.py \
  --llama_dir "$ROOT/ckpts/drivelm/llama_model_weights" \
  --checkpoint "$ROOT/ckpts/drivelm/BIAS-7B.pth" \
  --data "$ROOT/data/drivelm/test_llama_10.json" \
  --output "$OUT/test_output.json" \
  --batch_size 1 \
  --num_processes 1 \
  2>&1 | tee "$OUT/demo.log"
```

本轮使用官方 BIAS-7B + LLaMA-1，不是个人历史 finetune checkpoint。`10/10 非空` 只表示生成链路正常，不是 QA accuracy；答案仍需人工核对图像事实和幻觉。

## 9. OpenDriveVLA-0.5B

该模型复现依赖官方 gated checkpoint、UniAD vision tower、scene/track/map cache。当前仓库只保留 4 条 mini 可视化结果，没有发布一个与上游版本无关的通用 cache builder，因此定位为 **demo evidence**，不是一键 benchmark。

最低验收项：

```text
torch 2.1.2+cu118
mmcv 1.7.2
mmdet3d 1.0.0rc6
transformers 4.37.2
checkpoint: OpenDriveVLA/OpenDriveVLA-0.5B
output: JSONL，至少含 sample token、mission、预测 6×0.5s trajectory
```

mini 自建 cache 中历史轨迹可能为空；任何 L2 数字必须标注为样例级参考，不能写成官方评测结果。

## 10. 在线 3D MOT

### 检测结果回放

```bash
python scripts/online_lidar_mot.py \
  --root "$ROOT" \
  --results-pkl "$ROOT/outputs/centerpoint/results.pkl" \
  --out-dir "$ROOT/outputs/tracking/online" \
  --scene scene-0103

python scripts/online_dual_mot.py \
  --root "$ROOT" \
  --results-pkl "$ROOT/outputs/centerpoint/results.pkl" \
  --streampetr-json "$ROOT/outputs/streampetr-r50/results_nusc.json" \
  --out-dir "$ROOT/outputs/tracking/online" \
  --scene scene-0103
```

### 同进程双模型真推理

```bash
export AD_ROOT="$ROOT"
bash scripts/run_online_e2e_dual_mot.sh

# mini 全场景长跑
cd "$ROOT/repos/StreamPETR"
python /path/to/adas-perception-stack-repro/scripts/online_e2e_dual_mot.py \
  --root "$ROOT" \
  --scenes all \
  --mini-all \
  --max-frames 0 \
  --tag long_all
```

公开延迟 JSON 记录 10 scenes、成功处理并计时 402 帧：CenterPoint 208.5 ms、StreamPETR 625.67 ms、tracker 6.55 ms、总计 910.72 ms，约 1.1 FPS。该数字是 4090 + PyTorch 路径，不是车载部署预算。

## 11. 复现完成定义

一次复现只有同时满足下列条件才算完成：

1. 命令退出码为 0。
2. 保存完整 stdout/stderr 日志，而不是只保留末尾截图。
3. 保存 config、upstream commit、checkpoint SHA256 和 conda history。
4. 评测样本数与期望一致，结果按 `sample_token` 对齐。
5. 指标和可视化来自同一结果文件。
6. 清楚区分 mini、full-val、demo、样例级指标和闭环指标。
7. 任何 upstream patch 都保存为可审查 diff。

已发布结果的证据等级和缺失项见 [结果与证据清单](docs/09-results-and-evidence.md)。
