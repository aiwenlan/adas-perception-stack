# Online MOT / E2E input manifest

## Replay（pkl）

| 模式 | scenes | frames | 产出 |
|---|---|---:|---|
| L-only long_val | scene-0103, scene-0916（mini val） | 81 | `assets/tracking/tracking_online_lidar_long_val.mp4` |
| L+C long_val | 同上 | 81 | `assets/tracking/tracking_online_lidar_cam_long_val.mp4` |

检测输入：CenterPoint / StreamPETR 的 `results.pkl`（私有路径 `$AD_ROOT/artifacts/day1/...`，不进 Git）。

## E2E live

| 项 | 值 |
|---|---|
| scenes | mini 全部 10 个（见 `latency_summary_long_all.json`） |
| 成功计时帧 | 402 |
| 检测器 | 同进程 CenterPoint + StreamPETR GPU 推理 |
| 跟踪器 | DualTracker（KF + gated NN） |
| 延迟摘要 | `latency_summary_long_all.json`（mean total 910.72 ms ≈ 1.1 FPS） |

## 明确不做

- 不报告 nuScenes Tracking Challenge AMOTA 等官方 MOT 指标  
- 不把 4090 PyTorch 延迟当作车端实时性能  
