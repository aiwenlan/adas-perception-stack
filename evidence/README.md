# Reproduction evidence

本目录存放各模型的**可审查小型证据**（manifest、命令、环境快照、脱敏日志、`metrics.json`）。  
大型 `results.pkl`、权重和 nuScenes / Occ3D 数据**不进 Git**。

> 总表与指标一览：[INDEX.md](INDEX.md)  
> 证据等级定义：[docs/09-results-and-evidence.md](../docs/09-results-and-evidence.md)

## 当前状态

已按模型落好证据包（等级多为 **B**；DriveLM / OpenDriveVLA 为 **C**）。  
有指标摘要 + 可视化 + commit / ckpt SHA256（能拿到则写入），但仍缺完整 `results.pkl` 归档 → **不是等级 A**。

| 目录 | 模型 | 等级 | 关键数字 |
|---|---|:---:|---|
| [bevformer-small](bevformer-small/) | BEVFormer-small | B | mAP 0.3541 / NDS 0.3989 |
| [bevformer-tiny](bevformer-tiny/) | BEVFormer-tiny（对照） | B | mAP 0.2649 / NDS 0.3255 |
| [streampetr-r50](streampetr-r50/) | StreamPETR-R50 | B | mAP 0.4370 / NDS 0.4831 |
| [centerpoint](centerpoint/) | CenterPoint voxel0.1 | B | mAP 0.5196 / NDS 0.5494 |
| [bevfusion](bevfusion/) | BEVFusion | B | fusion 0.5727 / 0.5798 |
| [fbocc-r50](fbocc-r50/) | FB-OCC R50 | B | mIoU 31.12 |
| [vad-tiny](vad-tiny/) | VAD-tiny | B | Plan L2 0.405 / 0.637 / 0.913 |
| [maptr-tiny](maptr-tiny/) | MapTR-tiny | B | Chamfer mAP 0.7129 |
| [online-mot](online-mot/) | Online MOT / E2E live | B | 402 帧 · 910.72 ms |
| [drivelm](drivelm/) | DriveLM | C | 10/10 非空（≠ accuracy） |
| [opendrivevla-0.5b](opendrivevla-0.5b/) | OpenDriveVLA-0.5B | C | 4 条 demo |

## 目录约定

```text
evidence/
├── INDEX.md
├── README.md
└── <model>/
    ├── manifest.md       # 必读：口径、commit、SHA256、局限
    ├── metrics.json      # 结构化指标
    ├── command.txt       # 复现命令（路径用 $AD_ROOT）
    ├── environment.txt   # conda / torch / GPU 快照
    ├── eval.log          # 日志尾部（已脱敏）
    ├── upstream.patch    # 可选：本地改动
    └── …                 # 可选：latency JSON、样例 JSONL
```

`manifest.md` 至少覆盖：date / GPU / conda env / upstream URL+commit / checkpoint SHA256 / dataset+sample count / command / raw result 私有路径 / visualization / reported metrics / known limitations。

## 刷新证据

在已跑通评测的机器上：

```bash
export AD_ROOT=/path/to/autonomous_driving_upgrade
# 在本仓库根目录：
bash scripts/pack_evidence.sh
```


脱敏规则：绝对云路径 → `$AD_ROOT`；IP、token、`hf_` 密钥 → `REDACTED`。公开前请再扫一遍日志。
