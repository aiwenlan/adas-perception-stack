# 05 · 技术栈总览

![技术栈演进](../assets/stack_evolution.svg)

## 能力地图

```text
传统模块化 L2
  → CenterPoint / StreamPETR
    → BEVFormer
        ├ BEVFusion
        ├ MapTR
        └ FB-OCC
    → VAD / UniAD
    → DriveLM（VLM）
    → VLA（轨迹 / Action Grounding）
```

## 复现快照

| 模块 | 结果 | 图示 |
|---|---|---|
| BEVFormer-small | mAP **0.354** / NDS **0.399** | [六相机](../assets/det/bevformer_cams.png) |
| StreamPETR-R50 | 0.437 / 0.483 | [六相机](../assets/det/streampetr_cams.png) |
| CenterPoint | 0.520 / 0.549 | [六相机](../assets/det/centerpoint_cams.png) |
| BEVFusion | fusion **0.573 / 0.580** | [六相机](../assets/det/bevfusion_cams.png) |
| FB-OCC | **mIoU 31.12**（Occ3D mini） | [pred vs GT](../assets/fbocc/sample_1_pred_vs_gt.png) |
| VAD-tiny | plan L2 0.405 / 0.637 / 0.913 | [showcase](../assets/vad/vad_showcase.png) |
| MapTR-tiny | Chamfer mAP 0.713 | [vectors](../assets/maptr/maptr_showcase.png) |
| Tracking（概念） | GT instance ID 跨帧 | [ID strip](../assets/tracking/tracking_gt_id_strip.png) |
| Online MOT | CenterPoint ± StreamPETR → KF+NN | [L](../assets/tracking/tracking_online_lidar.mp4) / [L+C](../assets/tracking/tracking_online_lidar_cam.mp4) |
| DriveLM | Graph VQA | [QA card](../assets/drivelm/drivelm_qa_card.png) |
| OpenDriveVLA-0.5B | mini 轨迹推理 | [showcase](../assets/opendrivevla/opendrivevla_mini_showcase.png) |

```mermaid
flowchart LR
  BF[BEVFormer-small 0.35] --> SP[StreamPETR 0.44]
  SP --> CP[CenterPoint 0.52]
  CP --> FU[BEVFusion 0.57]
```

## UniAD（参考）

![UniAD](../assets/uniad_tasks.svg)

## 工程清单

### 标定 / 同步
多相机 + LiDAR 时空同步、外参漂移；BEV / OCC 对标定误差敏感。

### Radar → BEV
雨雾夜仍有价值；可投影进 BEV，不宜被纯视觉叙事抹掉。

### 部署
按域拆延迟预算；详见 [07 · 量产部署与加速](07-deployment-acceleration.md)：TRT / FP16 / INT8、并行感知、OCC/地图降频、VLM 慢通道；超时降分辨率 / 时序深度 / 稀疏度。

### 评测
Det mAP/NDS · OCC mIoU · Map Chamfer · Plan L2 / 碰撞（open-loop 局限）· VLM QA · 闭环接管率。

### 数据闭环 / 安全 / World Model
问题挖掘 → 标注 → 回归；ODD + 规则兜底；生成式 / 世界模型作为研究视野。

## 相关文档

- 检测与投影 → [01](01-bev-3d-detection.md) · [FAQ](faq.md)  
- BEVFormer 内部 → [02](02-bevformer-pipeline.md)  
- Occupancy → [03](03-occupancy.md)  
- E2E / 地图 / MOT → [04](04-e2e-map-tracking.md)  
- VLM / VLA → [05a](05a-drivelm.md) · [05b](05b-vla.md)  
- 数据采集 / 建库 / 训练 → [06](06-data-and-training.md)  
- 量产部署与加速 → [07](07-deployment-acceleration.md)  
- 算法/数据选型与 conda 矩阵 → [08](08-selection-and-env-matrix.md)
- 环境与数据 → [SETUP.md](../SETUP.md)
