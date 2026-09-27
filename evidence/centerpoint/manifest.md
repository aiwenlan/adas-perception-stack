# CenterPoint (voxel 0.1)

date: prior mini-val run
machine / GPU: RTX 4090 24GB
conda env: streampetr (mmdet3d in StreamPETR tree)
upstream: mmdetection3d via StreamPETR checkout; commit `95f64702306ccdb7a78889578b2a55b5deb35b2a`
checkpoint SHA256: `9061688e5f81adae87d28241143e2d33075f68908134264f0de8c901acf911d8`
dataset version: nuScenes v1.0-mini val (81)
exact command: see command.txt
reported metrics: mAP **0.5196** / NDS **0.5494**
known limitations: used as MOT lidar birth/update source; not nuScenes tracking challenge score
