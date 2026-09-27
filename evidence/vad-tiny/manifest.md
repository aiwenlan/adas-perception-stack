# VAD-tiny

date: prior day2 eval
machine / GPU: RTX 4090 24GB
conda env: vad (or project env)
upstream URL: https://github.com/hustvl/VAD
upstream commit: `1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8`
checkpoint SHA256: `430e8881f33bdb19f86557db00c2ccc77da22b3bc042ba7f41a9ef4c2b258a16`
dataset: VAD temporal mini-filtered infos
exact command: see command.txt
visualization: `assets/vad/vad_showcase.png` (ego-frame BEV; lidar extrinsics not applied for plot)
reported metrics: Plan L2 @1/2/3s **0.405 / 0.637 / 0.913**; det mAP/NDS 0.266/0.359 (ref)
known limitations: open-loop; mini-filtered ≠ full val; collision=0 must not be sold as closed-loop safety
