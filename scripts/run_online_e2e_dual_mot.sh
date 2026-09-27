#!/bin/bash
# Same-process live GPU infer: CenterPoint + StreamPETR → DualTracker + latency.
# Requires: streampetr env, data/weights under $ROOT (see SETUP.md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AD_ROOT:-${ROOT:-/cloud/cloud-ssd1/autonomous_driving_upgrade}}"
ART="${OUT_DIR:-$ROOT/artifacts/day3_track/online_e2e}"
mkdir -p "$ART"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate streampetr

# StreamPETR repo must be cwd for plugin imports + relative data links
cd "$ROOT/repos/StreamPETR"
export PYTHONPATH="$SCRIPT_DIR:$ROOT/repos/StreamPETR:$ROOT/repos/StreamPETR/mmdetection3d:${PYTHONPATH:-}"
export AD_ROOT="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MM3D=$ROOT/repos/StreamPETR/mmdetection3d
mkdir -p "$MM3D/data"
if [ ! -e "$MM3D/data/nuscenes" ]; then
  ln -sfn "$ROOT/data/nuscenes" "$MM3D/data/nuscenes"
fi
if [ ! -e "$ROOT/repos/StreamPETR/data/nuscenes" ]; then
  mkdir -p "$ROOT/repos/StreamPETR/data"
  ln -sfn "$ROOT/data/nuscenes" "$ROOT/repos/StreamPETR/data/nuscenes"
fi

echo "===== $(date) online_e2e_dual_mot =====" | tee "$ART/run.log"
python "$SCRIPT_DIR/online_e2e_dual_mot.py" \
  --root "$ROOT" \
  --scene scene-0103 \
  --max-frames 40 \
  --thr 0.35 \
  2>&1 | tee -a "$ART/run.log"
echo E2E_LAUNCH_DONE | tee -a "$ART/run.log"
