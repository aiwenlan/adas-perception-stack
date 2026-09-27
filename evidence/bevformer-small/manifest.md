# BEVFormer-small

date: collected from prior mini-val run (artifacts/day1/bevformer_small_test)
machine / GPU: RTX 4090 24GB (single card)
driver / nvcc: see environment.txt
conda env: bevformer
upstream URL: https://github.com/fundamentalvision/BEVFormer
upstream commit: `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`
checkpoint: `bevformer_small_epoch_24.pth` (official pretrained)
checkpoint SHA256: `4ebc5810201ca1e609c29452f07c08bc4d9c9ebfda5d7f8ee1034dd446274bc5`
dataset version: nuScenes v1.0-mini val
evaluated sample count: 81
exact command: see command.txt
exit code: 0 (successful eval run)
raw result location: `$AD_ROOT/artifacts/day1/bevformer_small_test/` (not in git; large pkl)
visualization source: `scripts/run_viz_proj_final.sh` / `scripts/viz_bevformer_small.sh` → `assets/det/bevformer_*`
reported metrics: mAP **0.3541** / NDS **0.3989**
known limitations: mini-val only; not paper full-val; evidence level B until full results.pkl is archived privately
