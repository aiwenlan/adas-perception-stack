# Third-party components and attribution

本仓库不分发上游模型代码、训练数据或预训练权重。复现时使用的第三方项目包括：

| 项目 | 上游 |
|---|---|
| BEVFormer | https://github.com/fundamentalvision/BEVFormer |
| StreamPETR | https://github.com/exiawsh/StreamPETR |
| MMDetection3D / CenterPoint | https://github.com/open-mmlab/mmdetection3d |
| BEVFusion | https://github.com/mit-han-lab/bevfusion |
| FB-BEV / FB-OCC | https://github.com/NVlabs/FB-BEV |
| VAD | https://github.com/hustvl/VAD |
| UniAD | https://github.com/OpenDriveLab/UniAD |
| MapTR | https://github.com/hustvl/MapTR |
| DriveLM | https://github.com/OpenDriveLab/DriveLM |
| 个人 DriveLM 适配仓库 | https://github.com/aiwenlan/Finetune-e2e-model-DriveLM |
| OpenDriveVLA | 以项目官方仓库与 `OpenDriveVLA/OpenDriveVLA-0.5B` 模型卡为准 |
| nuScenes | https://www.nuscenes.org/ |
| Occ3D | https://github.com/Tsinghua-MARS-Lab/Occ3D |

使用者必须分别遵守各项目、权重和数据集的许可证、访问条件与引用要求。本仓库中的指标不改变上游模型归属。

`scripts/online_lidar_mot.py`、`scripts/online_dual_mot.py` 与 `scripts/online_e2e_dual_mot.py` 包含针对本项目的在线跟踪与可视化实现，并借鉴作者既有的 KF + nearest-neighbor MOT 工程思路。若其中包含从第三方代码直接改写的片段，应在发布前补充对应文件级版权头和许可证文本。

## Repository license

本仓库**原创**文档与脚本采用根目录 [LICENSE](LICENSE)（**MIT**）。  
该许可**不覆盖**上表第三方项目、预训练权重与数据集；使用者须另行遵守其许可与访问条件。
