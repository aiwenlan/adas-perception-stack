# FB-OCC R50

date: prior Occ3D mini eval
machine / GPU: RTX 4090 24GB
conda env: fbocc
upstream URL: https://github.com/NVlabs/FB-BEV (FB-OCC configs)
upstream commit: `6e25469256d98e7fcb52cc43efe812dc2fd2b446`
checkpoint SHA256: `665becca4798e3b52d69749567c502f8e3e45949ca97d886709f19928033fa03`
dataset version: Occ3D-nuScenes aligned to mini; labels.npz count=404; eval samples=81
exact command: see command.txt
reported metrics: mIoU **31.12** (mask_camera)
visualization source: `scripts/run_occ_showcase_viz.sh` (display orientation fixed: Occ3D [x_fwd,y_left] → imshow [::-1,::-1])
known limitations: mini ≠ official full-val (~39); viz post-process does not change mIoU
