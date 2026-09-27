# Online 3D MOT / E2E live

date: mini long runs
machine / GPU: RTX 4090 24GB
conda env: streampetr (+ model envs for live)
detectors: CenterPoint (lidar) ± StreamPETR (camera); DualTracker
exact command: see command.txt
latency: see latency_summary_long_all.json (402 frames, mean total **910.72 ms** ≈ 1.1 FPS)
visualizations: `assets/tracking/*`
input manifest: see `scene_manifest.md`
script hashes: see `script_sha256.txt`
known limitations: no official MOT metrics; 4090 PyTorch latency ≠ vehicle real-time; replay ≠ challenge submission
