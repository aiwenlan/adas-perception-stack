# StreamPETR-R50

date: prior mini-val run
machine / GPU: RTX 4090 24GB
conda env: streampetr
upstream URL: https://github.com/exiawsh/StreamPETR
upstream commit: `95f64702306ccdb7a78889578b2a55b5deb35b2a`
checkpoint: `stream_petr_r50_flash_704_bs2_seq_90e.pth`
checkpoint SHA256: `e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451`
dataset version: nuScenes v1.0-mini val (81)
exact command: see command.txt
raw result location: `$AD_ROOT/artifacts/day1/streampetr_r50_test/`
visualization source: `scripts/run_viz_proj_final.sh` → `assets/det/streampetr_*`
reported metrics: mAP **0.4370** / NDS **0.4831**
known limitations: flash-attn may be stubbed to normal attn on some installs; see upstream.patch
