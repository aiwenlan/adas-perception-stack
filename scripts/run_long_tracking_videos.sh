#!/bin/bash
# Longer mini tracking videos (no full nuScenes download).
# 1) L-only + L+C replay on mini-val (81 frames, 2 scenes)
# 2) E2E live on all mini scenes (402 frames successfully processed/timed in the published run)
set -euo pipefail
ROOT=/cloud/cloud-ssd1/autonomous_driving_upgrade
SCRIPT_DIR="$ROOT/scripts"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate streampetr
export AD_ROOT="$ROOT"
export PYTHONPATH="$SCRIPT_DIR:$ROOT/repos/StreamPETR:$ROOT/repos/StreamPETR/mmdetection3d:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT/repos/StreamPETR"

LOG=$ROOT/artifacts/day3_track/long_run.log
mkdir -p "$ROOT/artifacts/day3_track"
echo "===== $(date) LONG TRACK START =====" | tee "$LOG"

echo "=== L-only val long ===" | tee -a "$LOG"
python "$SCRIPT_DIR/online_lidar_mot.py" \
  --root "$ROOT" --scenes val --max-frames 0 --tag long_val \
  2>&1 | tee -a "$LOG"

echo "=== L+C val long ===" | tee -a "$LOG"
python "$SCRIPT_DIR/online_dual_mot.py" \
  --root "$ROOT" --scenes val --max-frames 0 --tag long_val \
  2>&1 | tee -a "$LOG"

echo "=== E2E all mini live ===" | tee -a "$LOG"
python "$SCRIPT_DIR/online_e2e_dual_mot.py" \
  --root "$ROOT" --scenes all --max-frames 0 --mini-all --tag long_all \
  --warmup 2 --fps 5 \
  2>&1 | tee -a "$LOG"

echo "===== $(date) LONG TRACK DONE =====" | tee -a "$LOG"
ls -lh "$ROOT/notes/assets/tracking/"*long* "$ROOT/notes/assets/tracking/"tracking_*lidar*.mp4 2>/dev/null | tee -a "$LOG"
