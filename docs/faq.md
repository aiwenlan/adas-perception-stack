# FAQ — 投影与坐标系

把 nuScenes 3D 框投到相机图像、或送进 MOT 时的常见坑。

## 对齐

1. 按 `sample_token` 配帧，禁止按 list 下标硬配。
2. JSON（`results_nusc.json`）通常在 **global** → 直接投到各相机。
3. pkl（CenterPoint / BEVFusion）通常在 **LiDAR** → ego → cam。
4. mmdet3d `x_size / y_size` ↔ NuScenes Box `l / w` 必须互换，否则框会「隧道化」。

## z 约定（因模型而异）

同帧 `06be0e3b66` / scene-0916，与 GT 底边对齐；`dby < 0` 表示画出来的框在图像上偏高。


| 模型                  | 输出含义                              | 正确画法                            | vs GT（正确时）         | 画错会怎样                  |
| ------------------- | --------------------------------- | ------------------------------- | ------------------ | ---------------------- |
| **BEVFormer** JSON  | 已是 NuScenes **gravity**           | **原样；禁止 +h/2**                  | dZ≈+0.07m，dby≈−9px | `+h/2` → dby≈−29px（浮空） |
| **StreamPETR** JSON | 已是 NuScenes **gravity**           | **原样；禁止 +h/2**                  | dZ≈+0.07m，dby≈−5px | `+h/2` → dby≈−35px     |
| **CenterPoint** pkl | `tensor[:,2]` ≈ **bottom**（~6cm）  | `z+h/2` **或** `.gravity_center` | dZ≈+0.08m，dby≈−9px | 原样 → 沉到地里（dby≈+38px）   |
| **BEVFusion** pkl   | `tensor[:,2]` ≈ **gravity**（~2cm） | **原样；不要再** `.gravity_center`    | dZ≈0，dby≈0.6px     | `+h/2` 再抬 ~0.8m        |


残差 5–9px 是模型本身略偏高，**不是**再缺一次 `h/2`。消融图：  
[BEVFormer](../assets/det/bevformer_z_ablation.png) · [StreamPETR](../assets/det/streampetr_z_ablation.png) · [CenterPoint](../assets/det/centerpoint_z_ablation.png) · [BEVFusion](../assets/det/bevfusion_z_ablation.png)

## CenterPoint → MOT / NuScenes 的 yaw

对本仓库使用的 CenterPoint pkl：

- dims → NuScenes：`[y_size, x_size, z_size]`  
- **yaw 保持原值**  
- **不要**套用 BEVFusion 的 `yaw = -yaw - π/2`（BEV 上会整体转约 90°）

BEV 画框：长度沿 yaw（`±l/2` 前进方向，`±w/2` 横向）。

## BEV 俯视图「哪边是前」

车体 **ego**：`x` 前、`y` 左、`z` 上。  
注意：nuScenes 车上 **LIDAR_TOP 相对 ego 约有 −90° yaw**，`get_sample_data(LIDAR)` 的 x **不是**车头朝向。俯视图必须用 **ego**，不要停在 lidar 系。

本仓库 `*_bev.png`：


| 图上方向            | 车体             |
| --------------- | -------------- |
| **上（箭头 / FWD）** | **前** `+x_ego` |
| 右               | 右 `−y_ego`     |
| 左               | 左 `+y_ego`     |


同类约定也适用于：


| 资产                            | 注意                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------- |
| `*_bev.png` / VAD / MapTR BEV | 框/矢量先到 **ego**，再 `plot(-y, x)`；**不要**再乘 LIDAR 外参                             |
| Occ3D / FB-OCC 俯视图            | 体素是 `[x前, y左, z]`；`imshow` 前对 BEV 做 `[::-1, ::-1]`，否则后方会画在上方                 |
| OpenDriveVLA 轨迹               | cache 里是 `(lateral, forward)`，直接 `plot(x,y)` 即 ↑前；**不要**再套 nuScenes `(-y,x)` |
| MOT `render_bev`              | 已在世界系按 ego yaw 旋到车体，↑为前                                                      |


过时脚本（`run_viz_proj_v2`–`v5` 等）仍可能含 lidar-BEV，请以 `run_viz_proj_final.sh` / `rerender_*_ego.sh` 为准。

## 展示约定

- 展示 thr ≈ 0.35–0.45；每帧大约 15–20 框便于阅读。  
- 全量 thr，**禁止** GT-align 过滤美化。  
- mini + 预训练推理，勿对标论文 full-val 观感。

