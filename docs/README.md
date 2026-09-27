# 文档导航

本目录不按论文发布时间排列，而是按自动驾驶系统的数据流组织：空间感知 → 场景表示 → 时序/地图/规划 → VLM/VLA → 数据训练与部署。

## 三条阅读路线

### 路线 A：60 分钟建立全局认识

1. [05 · 技术栈总览](05-stack-overview.md)
2. [01 · BEV 与 3D 检测](01-bev-3d-detection.md)
3. [03 · Occupancy](03-occupancy.md)
4. [04 · 端到端、地图与跟踪](04-e2e-map-tracking.md)
5. [05a · DriveLM](05a-drivelm.md) → [05b · VLA](05b-vla.md)

完成标准：能解释 dense BEV、sparse query、voxel、OCC、map query、track query 和 planning trajectory 分别解决什么问题。

### 路线 B：代码与复现

1. [环境与数据](../SETUP.md)
2. [选型与环境矩阵](08-selection-and-env-matrix.md)
3. [复现命令](../REPRO.md)
4. [BEVFormer 数据流](02-bevformer-pipeline.md)
5. [FAQ：坐标与投影](faq.md)
6. [结果与证据清单](09-results-and-evidence.md)

完成标准：能说清使用的 config、checkpoint、数据 split、输出文件、评测入口、坐标系和证据缺口。

### 路线 C：训练与产品化

1. [数据采集、建库与训练](06-data-and-training.md)
2. [量产部署与加速](07-deployment-acceleration.md)
3. [端到端、地图与跟踪](04-e2e-map-tracking.md)
4. [结果与证据清单](09-results-and-evidence.md)

完成标准：能从数据、标注、训练、部署、ODD、fallback 和闭环验证角度评审一项模型，而不是只比较离线精度。

## 文档关系

```text
01 BEV/3D Detection ──→ 02 BEVFormer pipeline
          │
          ├───────────→ 03 Occupancy
          ├───────────→ 04 E2E / Map / Tracking
          └───────────→ 05 Stack overview
                              ├─ 05a DriveLM
                              └─ 05b VLA

06 Data & Training ──→ 07 Deployment ──→ 08 Selection/Environment
          └────────────────────────────→ 09 Evidence

FAQ supports 01–04 and REPRO.
```

## 术语与口径

| 术语 | 本仓库中的含义 |
|---|---|
| **跑通 / inference** | 官方预训练权重完成前向推理并产生结构化输出或可视化 |
| **evaluation** | 通过对应项目 evaluator 计算 mini split 指标 |
| **demo** | 少量样例证明链路可执行，不代表总体性能 |
| **E2E live MOT** | 两个检测器同进程逐帧推理后进入 tracker，不代表联合训练网络 |
| **mini** | nuScenes v1.0-mini 或从其派生的 Occ3D mini，不是 full trainval |
| **证据 A/B/C/D** | 结果可审查程度，定义见 09；不是模型性能评级 |

## 关键边界

- 文档中的模型数字只用于本项目复现自检，不用于跨论文排名。
- 架构理解、官方权重推理、个人微调、从头训练和量产部署必须明确区分。
- 可视化经过阈值筛选和绘图变换，但不能通过 GT 匹配过滤来美化预测。
- 任何规划结果都要区分 open-loop 与 closed-loop。
- 任何延迟结果都要同时说明硬件、框架、精度、是否包含预后处理和统计分位数。

返回 [项目首页](../README.md)。
