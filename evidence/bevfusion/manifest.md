# BEVFusion

date: prior mini-val run
machine / GPU: RTX 4090 24GB
conda env: bevfusion
upstream URL: https://github.com/mit-han-lab/bevfusion
upstream commit: `326653dc06e0938edf1aae7d01efcd158ba83de5`
checkpoint (fusion) SHA256: `ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122`
dataset version: nuScenes v1.0-mini val (81)
exact command: see command.txt
reported metrics: lidar-only mAP/NDS **0.5424 / 0.5655**; cam+lidar **0.5727 / 0.5798**
known limitations: detection-layer fusion reference only; MOT path uses CenterPoint+StreamPETR instead
