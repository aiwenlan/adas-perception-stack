# Scripts

| 脚本 | 作用 |
|---|---|
| `online_lidar_mot.py` | CenterPoint → KF + 最近邻跟踪（LiDAR；可接 pkl 回放） |
| `online_dual_mot.py` | CenterPoint birth/update + StreamPETR camera update（结果回放） |
| `online_e2e_dual_mot.py` | 同进程 CP+SP **GPU 真推理** → DualTracker，并记录 ms |
| `run_online_e2e_dual_mot.sh` | e2e 单 scene 一键启动 |
| `run_long_tracking_videos.sh` | mini-val 回放加长 + mini 全 scene E2E 加长 |
| `pack_evidence.sh` | 从训练机打包/刷新 `../evidence/<model>/`（脱敏 log、SHA256、manifest） |

可视化评测图（det / OCC / VAD / MapTR 等）的生成脚本在工作区根目录 `scripts/run_viz_*.sh`、`rerender_*.sh`（体积与路径依赖较大，未全部镜像进本目录）。

## MOT 用法

```bash
# 短 demo
python online_lidar_mot.py --root $AD_ROOT --scene scene-0103 --max-frames 40
python online_dual_mot.py --root $AD_ROOT --scenes scene-0103 --max-frames 40

# 加长（不下全量 nuScenes）
python online_lidar_mot.py --root $AD_ROOT --scenes val --max-frames 0 --tag long_val
python online_dual_mot.py --root $AD_ROOT --scenes val --max-frames 0 --tag long_val
python online_e2e_dual_mot.py --root $AD_ROOT --scenes all --mini-all --max-frames 0 --tag long_all

bash run_long_tracking_videos.sh
```

`--scenes`：`scene-0103,scene-0916` / `val` / `all`；`--max-frames 0` = 该 scene 全部 keyframe。

## 刷新 evidence

```bash
export AD_ROOT=/path/to/autonomous_driving_upgrade
bash pack_evidence.sh
```

算法对齐公开 Lidar-based-Multi-Object-Tracking（Kalman Filter + gated NN）。演示与延迟见 `../assets/tracking/`；证据见 `../evidence/online-mot/`。

完整环境见 [SETUP.md](../SETUP.md)、[REPRO.md](../REPRO.md)。
