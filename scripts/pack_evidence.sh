#!/usr/bin/env bash
# Pack sanitized reproduction evidence into <this-repo>/evidence/<model>/
# Usage: AD_ROOT=/path/to/workspace bash scripts/pack_evidence.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${AD_ROOT:-/cloud/cloud-ssd1/autonomous_driving_upgrade}"
OUT="$REPO_ROOT/evidence"
mkdir -p "$OUT"

sanitize() {
  # redact absolute cloud paths / home paths; keep structure readable
  sed -E \
    -e "s|$ROOT|\$AD_ROOT|g" \
    -e 's|/cloud/cloud-ssd[0-9]+/[^[:space:]]+|\$AD_ROOT/...|g' \
    -e 's|/home/[^/[:space:]]+|\$HOME|g' \
    -e 's|https://[^[:space:]]*token=[^[:space:]]+|https://REDACTED|g' \
    -e 's|hf_[A-Za-z0-9]{20,}|hf_REDACTED|g' \
    -e 's|[0-9]{1,3}(\.[0-9]{1,3}){3}|REDACTED_IP|g'
}

tail_sanitized() {
  local src="$1" dest="$2" n="${3:-200}"
  if [[ -f "$src" ]]; then
    tail -n "$n" "$src" | sanitize > "$dest"
  else
    echo "(missing: $(basename "$src"))" > "$dest"
  fi
}

extract_metrics_block() {
  local src="$1" dest="$2"
  if [[ ! -f "$src" ]]; then
    echo "{}" > "$dest"
    return
  fi
  # keep last metrics-ish section from log
  grep -E -i 'mAP|NDS|mIoU|Chamfer|Eval_Results|Overall|L2|collision|fps|mean' "$src" | tail -n 80 | sanitize > "${dest}.txt" || true
}

repo_commit() {
  local repo="$1"
  if [[ -d "$repo/.git" ]]; then
    git -C "$repo" rev-parse HEAD 2>/dev/null || echo "unknown"
  else
    echo "unknown"
  fi
}

ckpt_sha() {
  local f="$1"
  if [[ -f "$f" ]]; then
    sha256sum "$f" | awk '{print $1}'
  else
    echo "missing"
  fi
}

env_snapshot() {
  local envname="$1" dest="$2"
  {
    echo "conda_env: $envname"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname: REDACTED"
    echo "uname: $(uname -srm)"
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1 | sed 's/^/gpu: /'
    fi
    if command -v nvcc >/dev/null; then
      echo "nvcc: $(nvcc --version 2>/dev/null | tail -1)"
    fi
    source /usr/local/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    if conda activate "$envname" 2>/dev/null; then
      python - <<'PY'
import sys
print('python:', sys.version.split()[0])
try:
  import torch
  print('torch:', torch.__version__, 'cuda', getattr(torch.version,'cuda',None))
except Exception as e:
  print('torch:', e)
for m in ('mmcv','mmdet','mmdet3d','mmseg'):
  try:
    mod=__import__(m)
    print(f'{m}:', getattr(mod,'__version__','?'))
  except Exception:
    pass
PY
    else
      echo "conda activate $envname: failed"
    fi
  } > "$dest" 2>&1 || true
}

write_manifest() {
  local dir="$1"
  cat > "$dir/manifest.md"
}

# ---------- helpers per model ----------
pack_one() {
  local name="$1"
  local dir="$OUT/$name"
  mkdir -p "$dir"
  echo "=== packing $name ==="
}

# ============== BEVFormer-small ==============
pack_one bevformer-small
DIR="$OUT/bevformer-small"
REPO="$ROOT/repos/BEVFormer"
CKPT="$ROOT/ckpts/bevformer/bevformer_small_epoch_24.pth"
LOG="$ROOT/artifacts/day1/bevformer_small_test/test2.log"
[[ -f "$LOG" ]] || LOG="$ROOT/artifacts/day1/bevformer_small_test/test.log"
env_snapshot bevformer "$DIR/environment.txt"
tail_sanitized "$LOG" "$DIR/eval.log" 250
COMMIT=$(repo_commit "$REPO")
SHA=$(ckpt_sha "$CKPT")
cat > "$DIR/command.txt" <<'EOF'
conda activate bevformer
cd $AD_ROOT/repos/BEVFormer
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python tools/test.py \
  projects/configs/bevformer/bevformer_small.py \
  $AD_ROOT/ckpts/bevformer/bevformer_small_epoch_24.pth \
  --eval bbox \
  2>&1 | tee $AD_ROOT/outputs/bevformer-small/eval.log
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "BEVFormer-small",
  "dataset": "nuScenes v1.0-mini val",
  "sample_count": 81,
  "metrics": {"mAP": 0.3541, "NDS": 0.3989},
  "evidence_level": "B",
  "visualizations": [
    "assets/det/bevformer_cams.png",
    "assets/det/bevformer_bev.png",
    "assets/det/bevformer_gt_vs_pred.png"
  ]
}
EOF
write_manifest "$DIR" <<EOF
# BEVFormer-small

date: collected from prior mini-val run (artifacts/day1/bevformer_small_test)
machine / GPU: RTX 4090 24GB (single card)
driver / nvcc: see environment.txt
conda env: bevformer
upstream URL: https://github.com/fundamentalvision/BEVFormer
upstream commit: \`$COMMIT\`
checkpoint: \`bevformer_small_epoch_24.pth\` (official pretrained)
checkpoint SHA256: \`$SHA\`
dataset version: nuScenes v1.0-mini val
evaluated sample count: 81
exact command: see command.txt
exit code: 0 (successful eval run)
raw result location: \`\$AD_ROOT/artifacts/day1/bevformer_small_test/\` (not in git; large pkl)
visualization source: \`scripts/run_viz_proj_final.sh\` / \`scripts/viz_bevformer_small.sh\` → \`assets/det/bevformer_*\`
reported metrics: mAP **0.3541** / NDS **0.3989**
known limitations: mini-val only; not paper full-val; evidence level B until full results.pkl is archived privately
EOF

# ============== BEVFormer-tiny ==============
pack_one bevformer-tiny
DIR="$OUT/bevformer-tiny"
LOG="$ROOT/artifacts/day1/bevformer_tiny_test/test.log"
env_snapshot bevformer "$DIR/environment.txt"
tail_sanitized "$LOG" "$DIR/eval.log" 200
CKPT_T="$ROOT/ckpts/bevformer/bevformer_tiny_epoch_24.pth"
[[ -f "$CKPT_T" ]] || CKPT_T="$ROOT/ckpts/bevformer/bevformer_tiny.pth"
SHA=$(ckpt_sha "$CKPT_T")
COMMIT=$(repo_commit "$REPO")
cat > "$DIR/command.txt" <<'EOF'
conda activate bevformer
cd $AD_ROOT/repos/BEVFormer
python tools/test.py \
  projects/configs/bevformer/bevformer_tiny.py \
  $AD_ROOT/ckpts/bevformer/bevformer_tiny_epoch_24.pth \
  --eval bbox
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "BEVFormer-tiny",
  "dataset": "nuScenes v1.0-mini val",
  "sample_count": 81,
  "metrics": {"mAP": 0.2649, "NDS": 0.3255},
  "evidence_level": "B",
  "role": "lightweight control vs small; not primary showcase"
}
EOF
write_manifest "$DIR" <<EOF
# BEVFormer-tiny

date: prior mini-val control run
machine / GPU: RTX 4090 24GB
conda env: bevformer
upstream URL: https://github.com/fundamentalvision/BEVFormer
upstream commit: \`$COMMIT\`
checkpoint SHA256: \`$SHA\`
dataset version: nuScenes v1.0-mini val (81)
exact command: see command.txt
reported metrics: mAP **0.2649** / NDS **0.3255**
known limitations: control only; do not cite as primary BEVFormer result
EOF

# ============== StreamPETR ==============
pack_one streampetr-r50
DIR="$OUT/streampetr-r50"
REPO="$ROOT/repos/StreamPETR"
LOG="$ROOT/artifacts/day1/streampetr_r50_test/eval.log"
env_snapshot streampetr "$DIR/environment.txt"
tail_sanitized "$LOG" "$DIR/eval.log" 250
COMMIT=$(repo_commit "$REPO")
CKPT=$(ls "$ROOT"/ckpts/streampetr/*.pth 2>/dev/null | head -1 || true)
[[ -n "${CKPT:-}" ]] || CKPT=$(ls "$REPO"/ckpts/*.pth 2>/dev/null | head -1 || true)
SHA=$(ckpt_sha "${CKPT:-/nonexistent}")
# capture local patch if dirty
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" diff --stat > "$DIR/upstream.patch.stat" 2>/dev/null || true
  git -C "$REPO" diff -- 'tools/test.py' 'projects/**/*.py' 2>/dev/null | head -c 200000 | sanitize > "$DIR/upstream.patch" || echo "(no diff or too large)" > "$DIR/upstream.patch"
fi
cat > "$DIR/command.txt" <<'EOF'
conda activate streampetr
cd $AD_ROOT/repos/StreamPETR
export PYTHONPATH=$PWD:$PWD/mmdetection3d:${PYTHONPATH:-}
python tools/test.py \
  projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
  ckpts/stream_petr_r50_flash_704_bs2_seq_90e.pth \
  --eval bbox \
  --out $AD_ROOT/outputs/streampetr-r50/results.pkl
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "StreamPETR-R50",
  "dataset": "nuScenes v1.0-mini val",
  "sample_count": 81,
  "metrics": {"mAP": 0.4370, "NDS": 0.4831},
  "evidence_level": "B",
  "visualizations": [
    "assets/det/streampetr_cams.png",
    "assets/det/streampetr_bev.png",
    "assets/det/streampetr_gt_vs_pred.png"
  ]
}
EOF
write_manifest "$DIR" <<EOF
# StreamPETR-R50

date: prior mini-val run
machine / GPU: RTX 4090 24GB
conda env: streampetr
upstream URL: https://github.com/exiawsh/StreamPETR
upstream commit: \`$COMMIT\`
checkpoint: \`${CKPT##*/}\`
checkpoint SHA256: \`$SHA\`
dataset version: nuScenes v1.0-mini val (81)
exact command: see command.txt
raw result location: \`\$AD_ROOT/artifacts/day1/streampetr_r50_test/\`
visualization source: \`scripts/run_viz_proj_final.sh\` → \`assets/det/streampetr_*\`
reported metrics: mAP **0.4370** / NDS **0.4831**
known limitations: flash-attn may be stubbed to normal attn on some installs; see upstream.patch
EOF

# ============== CenterPoint ==============
pack_one centerpoint
DIR="$OUT/centerpoint"
REPO="$ROOT/repos/StreamPETR/mmdetection3d"
LOG="$ROOT/artifacts/day1/centerpoint_test/eval.log"
env_snapshot streampetr "$DIR/environment.txt"
tail_sanitized "$LOG" "$DIR/eval.log" 250
COMMIT=$(repo_commit "$ROOT/repos/StreamPETR")
CKPT=$(ls "$ROOT"/ckpts/centerpoint/*.pth 2>/dev/null | head -1 || true)
SHA=$(ckpt_sha "${CKPT:-/nonexistent}")
cat > "$DIR/command.txt" <<'EOF'
conda activate streampetr
cd $AD_ROOT/repos/StreamPETR/mmdetection3d
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python tools/test.py \
  configs/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py \
  $AD_ROOT/ckpts/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.pth \
  --eval bbox \
  --out $AD_ROOT/outputs/centerpoint/results.pkl
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "CenterPoint-voxel0.1",
  "dataset": "nuScenes v1.0-mini val",
  "sample_count": 81,
  "metrics": {"mAP": 0.5196, "NDS": 0.5494},
  "evidence_level": "B",
  "note": "OpenMMLab voxel 0.1; not BEVFusion lidar-only row",
  "visualizations": [
    "assets/det/centerpoint_cams.png",
    "assets/det/centerpoint_bev.png",
    "assets/det/centerpoint_gt_vs_pred.png"
  ]
}
EOF
write_manifest "$DIR" <<EOF
# CenterPoint (voxel 0.1)

date: prior mini-val run
machine / GPU: RTX 4090 24GB
conda env: streampetr (mmdet3d in StreamPETR tree)
upstream: mmdetection3d via StreamPETR checkout; commit \`$COMMIT\`
checkpoint SHA256: \`$SHA\`
dataset version: nuScenes v1.0-mini val (81)
exact command: see command.txt
reported metrics: mAP **0.5196** / NDS **0.5494**
known limitations: used as MOT lidar birth/update source; not nuScenes tracking challenge score
EOF

# ============== BEVFusion ==============
pack_one bevfusion
DIR="$OUT/bevfusion"
REPO="$ROOT/repos/BEVFusion"
env_snapshot bevfusion "$DIR/environment.txt"
tail_sanitized "$ROOT/artifacts/day1/bevfusion_test/eval.log" "$DIR/eval.log" 250
tail_sanitized "$ROOT/artifacts/day1/bevfusion_test/eval_lidar.log" "$DIR/eval_lidar.log" 200
COMMIT=$(repo_commit "$REPO")
CKPT=$(ls "$REPO"/pretrained/bevfusion-det.pth "$ROOT"/ckpts/bevfusion/*.pth 2>/dev/null | head -1 || true)
SHA=$(ckpt_sha "${CKPT:-/nonexistent}")
cat > "$DIR/command.txt" <<'EOF'
conda activate bevfusion
cd $AD_ROOT/repos/BEVFusion
# lidar-only control
python tools/test.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  pretrained/lidar-only-det.pth --eval bbox | tee eval-lidar.log
# camera+lidar
python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox | tee eval-fusion.log
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "BEVFusion",
  "dataset": "nuScenes v1.0-mini val",
  "sample_count": 81,
  "metrics": {
    "lidar_only": {"mAP": 0.5424, "NDS": 0.5655},
    "camera_lidar": {"mAP": 0.5727, "NDS": 0.5798}
  },
  "evidence_level": "B",
  "visualizations": [
    "assets/det/bevfusion_cams.png",
    "assets/det/bevfusion_bev.png",
    "assets/det/bevfusion_gt_vs_pred.png"
  ]
}
EOF
write_manifest "$DIR" <<EOF
# BEVFusion

date: prior mini-val run
machine / GPU: RTX 4090 24GB
conda env: bevfusion
upstream URL: https://github.com/mit-han-lab/bevfusion
upstream commit: \`$COMMIT\`
checkpoint (fusion) SHA256: \`$SHA\`
dataset version: nuScenes v1.0-mini val (81)
exact command: see command.txt
reported metrics: lidar-only mAP/NDS **0.5424 / 0.5655**; cam+lidar **0.5727 / 0.5798**
known limitations: detection-layer fusion reference only; MOT path uses CenterPoint+StreamPETR instead
EOF

# ============== FB-OCC ==============
pack_one fbocc-r50
DIR="$OUT/fbocc-r50"
REPO="$ROOT/repos/FB-BEV"
env_snapshot fbocc "$DIR/environment.txt"
LOG="$ROOT/artifacts/day1/fbocc_test/eval_miou.log"
[[ -f "$LOG" ]] || LOG="$ROOT/artifacts/day1/fbocc_test/eval.log"
tail_sanitized "$LOG" "$DIR/eval.log" 300
COMMIT=$(repo_commit "$REPO")
CKPT=$(ls "$ROOT"/ckpts/fbocc/*.pth 2>/dev/null | head -1 || true)
SHA=$(ckpt_sha "${CKPT:-/nonexistent}")
# GT count
N_GT=$(find "$ROOT/data/occ3d/gts" -name labels.npz 2>/dev/null | wc -l | tr -d ' ')
cat > "$DIR/command.txt" <<'EOF'
conda activate fbocc
cd $AD_ROOT/repos/FB-BEV
python tools/test.py \
  occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.py \
  $AD_ROOT/ckpts/fbocc/fbocc-r50-cbgs_depth_16f_16x4_20e.pth \
  --eval occupancy --launcher none \
  --out $AD_ROOT/outputs/fbocc/results_miou.pkl
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "FB-OCC-R50",
  "dataset": "Occ3D-nuScenes mini",
  "sample_count": 81,
  "gt_labels_npz_count": $N_GT,
  "metrics": {"mIoU": 31.12},
  "mask": "mask_camera",
  "evidence_level": "B",
  "visualizations": [
    "assets/fbocc/sample_1_pred_vs_gt.png",
    "assets/fbocc/sample_1_zslices.png",
    "assets/fbocc/sample_1_occ_only.png"
  ],
  "viz_note": "denoise / majority-Z / display flip are viz-only; mIoU from raw evaluator"
}
EOF
write_manifest "$DIR" <<EOF
# FB-OCC R50

date: prior Occ3D mini eval
machine / GPU: RTX 4090 24GB
conda env: fbocc
upstream URL: https://github.com/NVlabs/FB-BEV (FB-OCC configs)
upstream commit: \`$COMMIT\`
checkpoint SHA256: \`$SHA\`
dataset version: Occ3D-nuScenes aligned to mini; labels.npz count=$N_GT; eval samples=81
exact command: see command.txt
reported metrics: mIoU **31.12** (mask_camera)
visualization source: \`scripts/run_occ_showcase_viz.sh\` (display orientation fixed: Occ3D [x_fwd,y_left] → imshow [::-1,::-1])
known limitations: mini ≠ official full-val (~39); viz post-process does not change mIoU
EOF

# ============== VAD ==============
pack_one vad-tiny
DIR="$OUT/vad-tiny"
REPO="$ROOT/repos/VAD"
env_snapshot vad "$DIR/environment.txt" 2>/dev/null || env_snapshot streampetr "$DIR/environment.txt"
LOG=$(ls -t "$ROOT"/artifacts/day2/vad/*.log "$ROOT"/artifacts/day2/vad_run*_nohup.out 2>/dev/null | head -1 || true)
tail_sanitized "${LOG:-/nonexistent}" "$DIR/eval.log" 250
COMMIT=$(repo_commit "$REPO")
CKPT=$(ls "$ROOT"/ckpts/vad/*.pth "$REPO"/ckpts/*.pth 2>/dev/null | head -1 || true)
SHA=$(ckpt_sha "${CKPT:-/nonexistent}")
cat > "$DIR/command.txt" <<'EOF'
conda activate vad   # or project env used for VAD
cd $AD_ROOT/repos/VAD
python tools/test.py \
  projects/configs/VAD/VAD_tiny_e2e.py \
  $AD_ROOT/ckpts/vad/VAD_tiny.pth \
  --eval bbox \
  --out $AD_ROOT/outputs/vad/results.pkl
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "VAD-tiny",
  "dataset": "nuScenes mini-filtered (VAD temporal infos)",
  "metrics": {
    "plan_L2_1s": 0.405,
    "plan_L2_2s": 0.637,
    "plan_L2_3s": 0.913
  },
  "det_ref": {"mAP": 0.266, "NDS": 0.359},
  "evidence_level": "B",
  "visualizations": ["assets/vad/vad_showcase.png"],
  "note": "open-loop planning L2; not closed-loop safety"
}
EOF
write_manifest "$DIR" <<EOF
# VAD-tiny

date: prior day2 eval
machine / GPU: RTX 4090 24GB
conda env: vad (or project env)
upstream URL: https://github.com/hustvl/VAD
upstream commit: \`$COMMIT\`
checkpoint SHA256: \`$SHA\`
dataset: VAD temporal mini-filtered infos
exact command: see command.txt
visualization: \`assets/vad/vad_showcase.png\` (ego-frame BEV; lidar extrinsics not applied for plot)
reported metrics: Plan L2 @1/2/3s **0.405 / 0.637 / 0.913**; det mAP/NDS 0.266/0.359 (ref)
known limitations: open-loop; mini-filtered ≠ full val; collision=0 must not be sold as closed-loop safety
EOF

# ============== MapTR ==============
pack_one maptr-tiny
DIR="$OUT/maptr-tiny"
REPO="$ROOT/repos/MapTR"
env_snapshot maptr "$DIR/environment.txt" 2>/dev/null || env_snapshot streampetr "$DIR/environment.txt"
LOG=$(ls -t "$ROOT"/artifacts/day2/maptr/*.log "$ROOT"/artifacts/day2/maptr_run*_nohup.out 2>/dev/null | head -1 || true)
tail_sanitized "${LOG:-/nonexistent}" "$DIR/eval.log" 250
COMMIT=$(repo_commit "$REPO")
CKPT=$(ls "$ROOT"/ckpts/maptr/*.pth "$REPO"/ckpts/*.pth 2>/dev/null | head -1 || true)
SHA=$(ckpt_sha "${CKPT:-/nonexistent}")
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" diff -- 'tools/test.py' 2>/dev/null | head -c 200000 | sanitize > "$DIR/upstream.patch" || true
fi
cat > "$DIR/command.txt" <<'EOF'
conda activate maptr
cd $AD_ROOT/repos/MapTR
python tools/test.py \
  projects/configs/maptr/maptr_tiny_r50_24e.py \
  $AD_ROOT/ckpts/maptr/maptr_tiny_r50_24e.pth \
  --eval chamfer \
  --out $AD_ROOT/outputs/maptr/results.pkl
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "MapTR-tiny",
  "dataset": "nuScenes v1.0-mini",
  "metrics": {"chamfer_mAP": 0.7129},
  "evidence_level": "B",
  "visualizations": ["assets/maptr/maptr_showcase.png"],
  "viz_note": "vectors transformed LIDAR_TOP → ego before plot(-y,x)"
}
EOF
write_manifest "$DIR" <<EOF
# MapTR-tiny

date: prior day2 eval
machine / GPU: RTX 4090 24GB
conda env: maptr
upstream URL: https://github.com/hustvl/MapTR
upstream commit: \`$COMMIT\`
checkpoint SHA256: \`$SHA\`
exact command: see command.txt
reported metrics: Chamfer mAP **0.7129** (≈0.713 in tables)
visualization: \`assets/maptr/maptr_showcase.png\`
known limitations: mini; single-GPU test may need upstream.patch
EOF

# ============== DriveLM ==============
pack_one drivelm
DIR="$OUT/drivelm"
REPO="$ROOT/repos/DriveLM"
env_snapshot drivevla "$DIR/environment.txt" 2>/dev/null || echo "env: see SETUP.md DriveLM section" > "$DIR/environment.txt"
LOG="$ROOT/artifacts/day2/drivelm_llama1_demo.log"
[[ -f "$LOG" ]] || LOG="$ROOT/artifacts/day2/drivelm_demo.log"
tail_sanitized "$LOG" "$DIR/eval.log" 200
COMMIT=$(repo_commit "$REPO")
# copy small QA summary if present
if [[ -d "$ROOT/artifacts/day2/drivelm_qa" ]]; then
  find "$ROOT/artifacts/day2/drivelm_qa" -name '*.json' | head -3 | while read -r j; do
    sanitize < "$j" > "$DIR/$(basename "$j")" || true
  done
fi
cat > "$DIR/command.txt" <<'EOF'
# see REPRO.md DriveLM section — official BIAS-7B + LLaMA-1 demo
python demo.py --checkpoint $AD_ROOT/ckpts/drivelm/BIAS-7B.pth ...
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "DriveLM BIAS-7B",
  "dataset": "mini demo QA (10 prompts)",
  "metrics": {"non_empty_generations": "10/10"},
  "evidence_level": "C",
  "visualizations": ["assets/drivelm/drivelm_qa_card.png"],
  "warning": "10/10 non-empty is NOT accuracy"
}
EOF
write_manifest "$DIR" <<EOF
# DriveLM

date: prior demo
machine / GPU: RTX 4090 24GB
upstream commit: \`$COMMIT\`
checkpoint: official BIAS-7B (not personal finetune)
reported metrics: **10/10 non-empty generations** — not QA accuracy
visualization: \`assets/drivelm/drivelm_qa_card.png\`
known limitations: evidence level C; answers need human fact-check for hallucination
EOF

# ============== OpenDriveVLA ==============
pack_one opendrivevla-0.5b
DIR="$OUT/opendrivevla-0.5b"
REPO="$ROOT/repos/OpenDriveVLA"
env_snapshot drivevla "$DIR/environment.txt" 2>/dev/null || echo "env: drivevla" > "$DIR/environment.txt"
LOG="$ROOT/artifacts/day2/opendrivevla/build_cache_infer.log"
[[ -f "$LOG" ]] || LOG=$(ls -t "$ROOT"/artifacts/day2/opendrivevla/*.log 2>/dev/null | head -1 || true)
tail_sanitized "${LOG:-/nonexistent}" "$DIR/eval.log" 200
if [[ -f "$ROOT/artifacts/day2/opendrivevla/infer_mini/plan_conv_mini.jsonl" ]]; then
  sanitize < "$ROOT/artifacts/day2/opendrivevla/infer_mini/plan_conv_mini.jsonl" > "$DIR/plan_conv_mini.jsonl" || true
fi
COMMIT=$(repo_commit "$REPO")
cat > "$DIR/command.txt" <<'EOF'
# gated ckpt + UniAD tower + scene cache; see REPRO.md
# inference-only on 4 mini samples → viz cards
bash $AD_ROOT/scripts/viz_opendrivevla.sh
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "OpenDriveVLA-0.5B",
  "dataset": "nuScenes mini (4 samples)",
  "metrics": {"demo_samples": 4},
  "evidence_level": "C",
  "visualizations": ["assets/opendrivevla/opendrivevla_mini_showcase.png"],
  "traj_convention": "cache columns (lateral, forward); plot as-is for ↑forward"
}
EOF
write_manifest "$DIR" <<EOF
# OpenDriveVLA-0.5B

date: prior inference-only demo
machine / GPU: RTX 4090 24GB
upstream commit: \`$COMMIT\`
reported metrics: 4-sample demo only — **not a benchmark**
visualization: \`assets/opendrivevla/opendrivevla_mini_showcase.png\`
known limitations: gated weights; no public universal cache builder; evidence level C
EOF

# ============== Online MOT ==============
pack_one online-mot
DIR="$OUT/online-mot"
env_snapshot streampetr "$DIR/environment.txt"
cp -f "$REPO_ROOT/assets/tracking/latency_summary.json" "$DIR/" 2>/dev/null || true
cp -f "$REPO_ROOT/assets/tracking/latency_summary_long_all.json" "$DIR/" 2>/dev/null || true
# copy scripts hashes
{
  echo "online_lidar_mot.py sha256: $(sha256sum "$REPO_ROOT/scripts/online_lidar_mot.py" | awk '{print $1}')"
  echo "online_dual_mot.py sha256: $(sha256sum "$REPO_ROOT/scripts/online_dual_mot.py" | awk '{print $1}')"
  echo "online_e2e_dual_mot.py sha256: $(sha256sum "$REPO_ROOT/scripts/online_e2e_dual_mot.py" | awk '{print $1}')"
} > "$DIR/script_sha256.txt"
cat > "$DIR/command.txt" <<EOF
# replay (pkl)
python $REPO_ROOT/scripts/online_lidar_mot.py --scenes val --max-frames 0 --tag long_val
python $REPO_ROOT/scripts/online_dual_mot.py --scenes val --max-frames 0 --tag long_val
# live E2E
python $REPO_ROOT/scripts/online_e2e_dual_mot.py --scenes all --max-frames 0 --tag long_all
EOF
cat > "$DIR/metrics.json" <<EOF
{
  "model": "Online-3D-MOT + E2E-live",
  "dataset": "nuScenes mini",
  "replay": {"scenes": "mini-val", "frames": 81},
  "e2e_live": {
    "scenes": 10,
    "frames_timed": 402,
    "total_ms_mean": 910.72,
    "approx_fps": 1.1
  },
  "evidence_level": "B",
  "visualizations": [
    "assets/tracking/tracking_online_lidar.mp4",
    "assets/tracking/tracking_online_lidar_cam.mp4",
    "assets/tracking/tracking_e2e_live_long_all.mp4"
  ],
  "note": "custom KF+NN tracker; not nuScenes tracking challenge AMOTA"
}
EOF
write_manifest "$DIR" <<EOF
# Online 3D MOT / E2E live

date: mini long runs
machine / GPU: RTX 4090 24GB
conda env: streampetr (+ model envs for live)
detectors: CenterPoint (lidar) ± StreamPETR (camera); DualTracker
exact command: see command.txt
latency: see latency_summary_long_all.json (402 frames, mean total **910.72 ms** ≈ 1.1 FPS)
visualizations: \`assets/tracking/*\`
known limitations: no official MOT metrics; 4090 PyTorch latency ≠ vehicle real-time; replay ≠ challenge submission
EOF

# ============== index ==============
cat > "$OUT/INDEX.md" <<EOF
# Evidence index

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

| Folder | Model | Level | Key metrics |
|---|---|:---:|---|
| [bevformer-small](bevformer-small/) | BEVFormer-small | B | mAP 0.3541 / NDS 0.3989 |
| [bevformer-tiny](bevformer-tiny/) | BEVFormer-tiny (control) | B | mAP 0.2649 / NDS 0.3255 |
| [streampetr-r50](streampetr-r50/) | StreamPETR-R50 | B | mAP 0.4370 / NDS 0.4831 |
| [centerpoint](centerpoint/) | CenterPoint voxel0.1 | B | mAP 0.5196 / NDS 0.5494 |
| [bevfusion](bevfusion/) | BEVFusion | B | fusion 0.5727 / 0.5798 |
| [fbocc-r50](fbocc-r50/) | FB-OCC R50 | B | mIoU 31.12 |
| [vad-tiny](vad-tiny/) | VAD-tiny | B | Plan L2 0.405/0.637/0.913 |
| [maptr-tiny](maptr-tiny/) | MapTR-tiny | B | Chamfer mAP 0.7129 |
| [drivelm](drivelm/) | DriveLM | C | 10/10 non-empty |
| [opendrivevla-0.5b](opendrivevla-0.5b/) | OpenDriveVLA-0.5B | C | 4-sample demo |
| [online-mot](online-mot/) | Online MOT / E2E live | B | 402f · 910.72 ms |

Paths in logs use \`\$AD_ROOT\` placeholder. Large \`results.pkl\` / checkpoints stay outside git.

See also: [docs/09-results-and-evidence.md](../docs/09-results-and-evidence.md).
EOF

# Keep hand-authored evidence/README.md; only refresh INDEX.md above.
echo "DONE pack_evidence → $OUT"

find "$OUT" -maxdepth 2 -type f | sort | wc -l
ls -la "$OUT"
