# 结果与证据清单

这份清单用于区分“项目作者已经运行过”“公开仓库中可以看到结果”和“第三方可以独立复现”。三者不是一回事。

## 证据等级

| 等级 | 含义 |
|---|---|
| **A** | 命令、完整日志、config、checkpoint SHA256、upstream commit、**完整**原始 results 与可视化均可审查 |
| **B** | 有指标摘要、结果图、小型 evidence 包（manifest / 脱敏 log / SHA256 等），但完整 `results.pkl` 等大文件不在 Git |
| **C** | 只有 demo 截图/少量样例，不能视为正式评测 |
| **D** | 只读架构或文字分析，没有宣称运行 |

## 当前公开状态

| 模块 | 公开结论 | 等级 | 已在仓库 | 仍缺（故非 A） |
|---|---|:---:|---|---|
| BEVFormer-small | mini mAP 0.3541 / NDS 0.3989 | B | [evidence](../evidence/bevformer-small/) + 六相机/BEV/GT-vs-pred | 完整 results.pkl |
| BEVFormer-tiny | mini mAP 0.2649 / NDS 0.3255 | B | [evidence](../evidence/bevformer-tiny/)（对照） | 完整 results.pkl |
| StreamPETR-R50 | mini mAP 0.4370 / NDS 0.4831 | B | [evidence](../evidence/streampetr-r50/) + 投影图 + patch | 完整 results.pkl |
| CenterPoint voxel 0.1 | mini mAP 0.5196 / NDS 0.5494 | B | [evidence](../evidence/centerpoint/) + 投影图 | 完整 results.pkl |
| BEVFusion cam+lidar | mini mAP 0.5727 / NDS 0.5798 | B | [evidence](../evidence/bevfusion/)（含 lidar-only log） | 完整 results.pkl |
| FB-OCC R50 | Occ3D mini mIoU 31.12 | B | [evidence](../evidence/fbocc-r50/) + pred/GT 图 | 完整 results_miou.pkl |
| VAD-tiny | Plan L2 0.405 / 0.637 / 0.913 | B | [evidence](../evidence/vad-tiny/) + showcase | 完整 results.pkl；筛选样本清单 |
| MapTR-tiny | Chamfer mAP 0.7129 | B | [evidence](../evidence/maptr-tiny/) + 矢量图 | 完整 results.pkl |
| Online MOT / E2E live | 402 帧 · 910.72 ms | B | [evidence](../evidence/online-mot/) + 视频 + latency JSON | 官方 MOT 指标；逐帧 timing 全表 |
| DriveLM | 10/10 非空生成 | C | [evidence](../evidence/drivelm/) + QA card | 人工事实核验；≠ accuracy |
| OpenDriveVLA-0.5B | 4 条 mini 样例 | C | [evidence](../evidence/opendrivevla-0.5b/) + showcase | 通用 cache builder；正式评测；闭环 |
| UniAD | 任务关系图 | D | 架构/文字 | 本计划不要求运行 |

## 指标使用规则

1. 所有检测、地图、规划和 OCC 数字必须带 **nuScenes mini / Occ3D mini** 限定。
2. 不把 mini 数字与论文 full-val 表格作性能优劣结论。
3. 不把 `10/10 非空` 称为 DriveLM accuracy。
4. 不把 4 条 OpenDriveVLA 样例称为 benchmark。
5. 不把 4090 PyTorch 延迟称为车端实时性能。
6. 不把 open-loop planning L2 或 mini collision=0 称为闭环安全。
7. 图片必须由同一份结果文件按 `sample_token` 对齐生成，不允许 GT 匹配过滤后再展示。

## 本仓库 evidence 目录

入口：[evidence/README.md](../evidence/README.md) · 总表：[evidence/INDEX.md](../evidence/INDEX.md)

每个 `evidence/<model>/` 通常包含：

| 文件 | 内容 |
|---|---|
| `manifest.md` | 日期、环境、upstream commit、ckpt SHA256、口径与局限 |
| `metrics.json` | 结构化指标 |
| `command.txt` | 复现命令（`$AD_ROOT`） |
| `environment.txt` | conda / torch / GPU 快照 |
| `eval.log` | 日志尾部（已脱敏） |
| `upstream.patch` | 如有本地改动 |

刷新（训练机）：

```bash
# 在本仓库内执行（推荐）
bash scripts/pack_evidence.sh
# 若 monorepo 把本仓放在 $AD_ROOT 下：
# bash $AD_ROOT/adas-perception-stack-repro/scripts/pack_evidence.sh
```


仍不进 Git：完整 `results.pkl`、权重、Occ3D/nuScenes 数据 → 等级保持 **B/C**，不能自称 **A**。

本仓库定位为“有真实推理产物的工程学习与作品集项目”，不是一键可复现 full-val benchmark。
