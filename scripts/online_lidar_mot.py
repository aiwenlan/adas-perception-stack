#!/usr/bin/env python3
"""
Online L-only 3D MOT: feed CenterPoint detections frame-by-frame into a
Python port of 峰华 Lidar-based-Multi-Object-Tracking (KF + NN gating).

Mode:
  --source pkl   : temporal replay of results.pkl (pseudo-online; default)
  --source infer : reserved (load model; not required for MVP)

Output: BEV mp4 under artifacts/day3_track/online/
"""
from __future__ import annotations

import argparse
import colorsys
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

# ---- constants mirrored from track.cc / tracker.cc ----
K_INIT_POS = 1.0
K_INIT_VEL = 4.0
K_INIT_ACC = 1.0
K_INIT_YAW = 0.01
K_INIT_YAWR = 0.01
K_PRED_POS = 0.25
K_PRED_VEL = 1.0
K_PRED_ACC = 0.25
K_PRED_YAW = 0.01
K_PRED_YAWR = 0.01
K_LIDAR_POS_NOISE = 0.25
K_LIDAR_YAW_NOISE = 0.01
K_SMOOTH = 0.7
K_RELIABLE_RANGE = 50.0
K_GATE = 3.0
NUSC_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]
VEH = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle'}


@dataclass
class Detection3D:
    det_id: int
    timestamp: float
    xyz: np.ndarray  # (3,)
    size: np.ndarray  # (3,) l,w,h
    yaw: float
    score: float = 1.0
    category: str = 'car'
    distance_to_ego: float = -1.0


@dataclass
class SensorFrame:
    index: int
    timestamp: float
    ego_xy: np.ndarray  # (2,)
    ego_yaw: float
    detections: List[Detection3D]
    sample_token: str = ''


class KalmanFilter:
    def __init__(self, state: np.ndarray, cov: np.ndarray):
        self.state = state.astype(np.float64).copy()
        self.cov = cov.astype(np.float64).copy()

    def predict(self, F: np.ndarray, Q: np.ndarray):
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q

    def update(self, z: np.ndarray, R: np.ndarray, H: np.ndarray):
        y = z - H @ self.state
        S = H @ self.cov @ H.T + R
        S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-6)
        K = self.cov @ H.T @ S_inv
        self.state = self.state + K @ y
        self.cov = self.cov - K @ H @ self.cov


class Track:
    _next_id = 1

    def __init__(self, det: Detection3D):
        self.id = Track._next_id
        Track._next_id += 1
        self.created_ts = det.timestamp
        self.timestamp = det.timestamp
        self.last_update_ts = det.timestamp
        self.position = det.xyz.astype(np.float64).copy()
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.size = det.size.astype(np.float64).copy()
        self.yaw = float(det.yaw)
        self.yaw_rate = 0.0
        self.associated_lidar = 1
        self.category = det.category

        state = np.array([
            self.position[0], self.position[1], self.position[2],
            0, 0, 0, 0, 0, 0,
        ], dtype=np.float64)
        cov = np.zeros((9, 9))
        cov[0, 0] = cov[1, 1] = cov[2, 2] = K_INIT_POS
        cov[3, 3] = cov[4, 4] = cov[5, 5] = K_INIT_VEL
        cov[6, 6] = cov[7, 7] = cov[8, 8] = K_INIT_ACC
        self.motion_kf = KalmanFilter(state, cov)

        yaw_state = np.array([self.yaw, self.yaw_rate], dtype=np.float64)
        yaw_cov = np.diag([K_INIT_YAW, K_INIT_YAWR])
        self.yaw_kf = KalmanFilter(yaw_state, yaw_cov)

    def predict(self, timestamp: float):
        if timestamp <= self.timestamp:
            return
        dt = timestamp - self.timestamp
        dt2 = dt * dt
        self.timestamp = timestamp
        F = np.eye(9)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        F[0, 6] = F[1, 7] = F[2, 8] = dt2
        F[3, 6] = F[4, 7] = F[5, 8] = dt
        Q = np.zeros((9, 9))
        Q[0, 0] = Q[1, 1] = Q[2, 2] = K_PRED_POS
        Q[3, 3] = Q[4, 4] = Q[5, 5] = K_PRED_VEL
        Q[6, 6] = Q[7, 7] = Q[8, 8] = K_PRED_ACC
        self.motion_kf.predict(F, Q)
        self.position = self.motion_kf.state[:3].copy()
        self.velocity = self.motion_kf.state[3:6].copy()
        self.acceleration = self.motion_kf.state[6:9].copy()

        Fy = np.array([[1, dt], [0, 1]], dtype=np.float64)
        Qy = np.diag([K_PRED_YAW, K_PRED_YAWR])
        self.yaw_kf.predict(Fy, Qy)
        self.yaw = float(self.yaw_kf.state[0])
        self.yaw_rate = float(self.yaw_kf.state[1])

    def update_lidar(self, det: Detection3D):
        z = det.xyz.astype(np.float64)
        scale = 1.0 if (det.distance_to_ego < 0 or det.distance_to_ego < K_RELIABLE_RANGE) else 2.0
        R = scale * K_LIDAR_POS_NOISE * np.eye(3)
        H = np.zeros((3, 9)); H[:, :3] = np.eye(3)
        self.motion_kf.update(z, R, H)
        self.position = self.motion_kf.state[:3].copy()
        self.velocity = self.motion_kf.state[3:6].copy()
        self.acceleration = self.motion_kf.state[6:9].copy()

        Ry = np.array([[scale * K_LIDAR_YAW_NOISE]])
        Hy = np.array([[1.0, 0.0]])
        self.yaw_kf.update(np.array([det.yaw]), Ry, Hy)
        self.yaw = float(self.yaw_kf.state[0])
        self.yaw_rate = float(self.yaw_kf.state[1])
        self.size = self.size * K_SMOOTH + det.size * (1 - K_SMOOTH)
        self.last_update_ts = det.timestamp
        self.associated_lidar += 1
        self.category = det.category

    def is_confirmed(self) -> bool:
        return self.associated_lidar > 2

    def is_lost(self) -> bool:
        gate = 1.5 if self.is_confirmed() else 0.1
        return self.timestamp - self.last_update_ts > gate

    def corners_bev(self) -> np.ndarray:
        # size stored as (l, w, h). yaw=0 → length along +x (forward).
        # Previous bug used (w/2, l/2) offsets → boxes drew sideways (横过来).
        theta = self.yaw
        c, s = np.cos(theta), np.sin(theta)
        def rot(p):
            # rotate local (x_forward, y_left) into world
            return np.array([p[0] * c - p[1] * s, p[0] * s + p[1] * c])
        center = self.position[:2]
        l, w = float(self.size[0]), float(self.size[1])
        shifts = [
            np.array([0.5 * l, 0.5 * w]),
            np.array([0.5 * l, -0.5 * w]),
            np.array([-0.5 * l, -0.5 * w]),
            np.array([-0.5 * l, 0.5 * w]),
        ]
        return np.stack([center + rot(sh) for sh in shifts], 0)


class Tracker:
    def __init__(self):
        self.tracks: List[Track] = []

    def predict_all(self, timestamp: float):
        for t in self.tracks:
            t.predict(timestamp)

    def run(self, frame: SensorFrame) -> List[Track]:
        self.predict_all(frame.timestamp)
        dets = frame.detections
        assoc: List[Tuple[int, int]] = []
        unassoc_det: List[int] = []
        used_track = [False] * len(self.tracks)

        if not dets:
            pass
        elif not self.tracks:
            unassoc_det = list(range(len(dets)))
        else:
            for i, d in enumerate(dets):
                best_j, best_dist = -1, K_GATE
                for j, tr in enumerate(self.tracks):
                    if used_track[j]:
                        continue
                    dist = float(np.linalg.norm(d.xyz - tr.position))
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                # mirror C++ bug-compatible: nearest_track_index > 0 (skips index 0)
                # Fix for online demo: use >= 0
                if best_j >= 0:
                    assoc.append((best_j, i))
                    used_track[best_j] = True
                else:
                    unassoc_det.append(i)

        for ti, di in assoc:
            self.tracks[ti].update_lidar(dets[di])

        for di in unassoc_det:
            self.tracks.append(Track(dets[di]))

        self.tracks = [t for t in self.tracks if not t.is_lost()]
        return [t for t in self.tracks if t.is_confirmed()]


def lidar_box_to_global(box7, sample, nusc) -> Box:
    """CenterPoint LiDAR box → NuScenes Box in global.

    本机对齐 GT：dims 用 [y_size, x_size, z_size]（→ NuScenes wlh），yaw 保持原值。
    不要用 BEVFusion 的 `yaw = -yaw - pi/2`（对本机 CenterPoint pkl 会整体偏 90°）。
    tensor z 为 bottom → gravity = z + h/2。
    """
    x, y, z, xs, ys, zs, yaw = [float(v) for v in box7[:7]]
    z = z + zs / 2.0
    box = Box([x, y, z], [ys, xs, zs], Quaternion(axis=[0, 0, 1], radians=yaw))
    lid_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs = nusc.get('calibrated_sensor', lid_sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', lid_sd['ego_pose_token'])
    box.rotate(Quaternion(cs['rotation'])); box.translate(np.array(cs['translation']))
    box.rotate(Quaternion(pose['rotation'])); box.translate(np.array(pose['translation']))
    return box


def load_centerpoint_map(root: Path) -> Dict[str, dict]:
    infos = pickle.load(open(root / 'data/nuscenes/nuscenes_infos_val.pkl', 'rb'))
    if isinstance(infos, dict) and 'infos' in infos:
        infos = infos['infos']
    results = pickle.load(open(root / 'artifacts/day1/centerpoint_test/results.pkl', 'rb'))
    return {i['token']: r for i, r in zip(infos, results)}


def frame_from_centerpoint(nusc, sample, res, thr=0.35, maxn=40) -> SensorFrame:
    if isinstance(res, dict) and 'pts_bbox' in res:
        res = res['pts_bbox']
    b = res['boxes_3d']
    b = b.tensor.cpu().numpy() if hasattr(b, 'tensor') else np.asarray(b)
    s = res['scores_3d']
    s = s.cpu().numpy() if hasattr(s, 'cpu') else np.asarray(s)
    lab = res['labels_3d']
    lab = lab.cpu().numpy() if hasattr(lab, 'cpu') else np.asarray(lab)

    lid_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose', lid_sd['ego_pose_token'])
    ego_xy = np.array(pose['translation'][:2], dtype=np.float64)
    # yaw from quaternion
    q = Quaternion(pose['rotation'])
    ego_yaw = q.yaw_pitch_roll[0]
    ts = sample['timestamp'] * 1e-6

    items = []
    for j in range(len(b)):
        if s[j] < thr:
            continue
        cname = NUSC_CLASSES[int(lab[j])] if 0 <= int(lab[j]) < len(NUSC_CLASSES) else 'car'
        if cname not in VEH:
            continue
        items.append((float(s[j]), b[j], int(lab[j]), cname))
    items.sort(key=lambda x: -x[0])
    items = items[:maxn]

    dets: List[Detection3D] = []
    for k, (sc, b7, lb, cname) in enumerate(items):
        gbox = lidar_box_to_global(b7, sample, nusc)
        xyz = np.array(gbox.center, dtype=np.float64)
        # NuScenes Box size is w,l,h — store as l,w,h for tracker
        w, l, h = gbox.wlh
        size = np.array([l, w, h], dtype=np.float64)
        yaw = gbox.orientation.yaw_pitch_roll[0]
        dist = float(np.linalg.norm(xyz[:2] - ego_xy))
        dets.append(Detection3D(k, ts, xyz, size, yaw, sc, cname, dist))

    return SensorFrame(
        index=0, timestamp=ts, ego_xy=ego_xy, ego_yaw=ego_yaw,
        detections=dets, sample_token=sample['token'],
    )


def color_for_id(tid: int):
    h = (tid * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def rot_ego(yaw, p):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])


def render_bev(frame: SensorFrame, tracks: List[Track], title: str, size=720) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (16, 20, 32)
    scale = size / 80.0  # ±80m

    def world_to_pix(xy):
        dx = xy[0] - frame.ego_xy[0]
        dy = xy[1] - frame.ego_xy[1]
        c, s = np.cos(-frame.ego_yaw), np.sin(-frame.ego_yaw)
        x = c * dx - s * dy  # forward
        y = s * dx + c * dy  # left
        u = int(size / 2 - y * scale)
        v = int(size / 2 - x * scale)
        return u, v

    for m in range(-40, 41, 10):
        pts_h = [world_to_pix(frame.ego_xy + rot_ego(frame.ego_yaw, np.array([m, y]))) for y in (-40.0, 40.0)]
        pts_v = [world_to_pix(frame.ego_xy + rot_ego(frame.ego_yaw, np.array([x, m]))) for x in (-40.0, 40.0)]
        cv2.line(img, pts_h[0], pts_h[1], (30, 40, 60), 1)
        cv2.line(img, pts_v[0], pts_v[1], (30, 40, 60), 1)

    ego_pts = np.array([
        world_to_pix(frame.ego_xy + rot_ego(frame.ego_yaw, np.array([2.0, 0.0]))),
        world_to_pix(frame.ego_xy + rot_ego(frame.ego_yaw, np.array([-1.0, 1.0]))),
        world_to_pix(frame.ego_xy + rot_ego(frame.ego_yaw, np.array([-1.0, -1.0]))),
    ], np.int32)
    cv2.fillConvexPoly(img, ego_pts, (220, 220, 220))

    for d in frame.detections:
        u, v = world_to_pix(d.xyz[:2])
        cv2.circle(img, (u, v), 3, (80, 80, 80), -1)

    for tr in tracks:
        col = color_for_id(tr.id)
        corners = tr.corners_bev()
        pts = np.array([world_to_pix(c) for c in corners], np.int32)
        cv2.polylines(img, [pts], True, col, 2)
        u, v = world_to_pix(tr.position[:2])
        cv2.putText(img, str(tr.id), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.rectangle(img, (0, 0), (size - 1, 48), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    return img


def pick_scene(nusc, name: Optional[str] = None):
    if name:
        for sc in nusc.scene:
            if sc['name'] == name:
                return sc
    # busiest
    best, best_n = None, -1
    for sc in nusc.scene:
        tok = sc['first_sample_token']
        n = 0
        while tok:
            s = nusc.get('sample', tok)
            n += len(s['anns'])
            tok = s['next']
        if n > best_n:
            best_n, best = n, sc
    return best


def resolve_scenes(nusc, scenes_arg: str = '', scene_arg: str = ''):
    """Parse --scenes (comma / 'all' / 'val') and legacy --scene."""
    raw = (scenes_arg or scene_arg or '').strip()
    if not raw or raw.lower() in ('auto',):
        return [pick_scene(nusc, None)]
    if raw.lower() == 'all':
        return list(nusc.scene)
    if raw.lower() == 'val':
        # nuScenes mini val scenes used in this repo
        want = {'scene-0103', 'scene-0916'}
        out = [sc for sc in nusc.scene if sc['name'] in want]
        return out or [pick_scene(nusc, None)]
    names = [x.strip() for x in raw.split(',') if x.strip()]
    out = []
    for name in names:
        sc = pick_scene(nusc, name)
        if sc is None or (name and sc['name'] != name):
            raise SystemExit(f'unknown scene: {name}')
        out.append(sc)
    return out


def scene_sample_chain(nusc, scene, max_frames: int = 0):
    chain = []
    tok = scene['first_sample_token']
    while tok:
        chain.append(nusc.get('sample', tok))
        tok = chain[-1]['next']
    if max_frames and max_frames > 0:
        chain = chain[:max_frames]
    return chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/cloud/cloud-ssd1/autonomous_driving_upgrade')
    ap.add_argument('--scene', default='', help='legacy single scene')
    ap.add_argument('--scenes', default='', help="comma list | 'val' | 'all'")
    ap.add_argument('--thr', type=float, default=0.35)
    ap.add_argument('--fps', type=float, default=5.0)
    ap.add_argument('--max-frames', type=int, default=0, help='per scene; 0 = all frames')
    ap.add_argument('--tag', default='long', help='output name tag')
    args = ap.parse_args()

    root = Path(args.root)
    out = root / 'artifacts/day3_track/online'
    notes = root / 'notes/assets/tracking'
    out.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version='v1.0-mini', dataroot=str(root / 'data/nuscenes'), verbose=False)
    cp_map = load_centerpoint_map(root)
    scenes = resolve_scenes(nusc, args.scenes, args.scene)
    print('scenes', [s['name'] for s in scenes], 'max_frames', args.max_frames)

    Track._next_id = 1
    tracker = Tracker()

    frames_img = []
    global_i = 0
    skipped = 0
    for scene in scenes:
        chain = scene_sample_chain(nusc, scene, args.max_frames)
        # soft reset association across scene cuts, keep ID counter monotonic
        tracker = Tracker()
        for i, sample in enumerate(chain):
            res = cp_map.get(sample['token'])
            if res is None:
                skipped += 1
                continue
            frame = frame_from_centerpoint(nusc, sample, res, thr=args.thr)
            frame.index = global_i
            published = tracker.run(frame)
            title = (f'ONLINE L-only | CenterPoint→峰华MOT | {scene["name"]} '
                     f'f={i+1}/{len(chain)} g={global_i+1} dets={len(frame.detections)} '
                     f'tracks={len(published)} | mini')
            frames_img.append(render_bev(frame, published, title))
            print(f'{scene["name"]} f{i}: dets={len(frame.detections)} confirmed={len(published)}')
            global_i += 1

    if not frames_img:
        raise SystemExit(f'no frames (skipped={skipped}; need CP results covering tokens)')

    h, w = frames_img[0].shape[:2]
    tag = args.tag or 'long'
    mp4 = out / f'tracking_online_lidar_{tag}.mp4'
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    for im in frames_img:
        vw.write(im)
    vw.release()
    import shutil
    shutil.copy2(mp4, out / 'tracking_online_lidar.mp4')
    shutil.copy2(mp4, notes / 'tracking_online_lidar.mp4')
    shutil.copy2(mp4, notes / f'tracking_online_lidar_{tag}.mp4')
    cv2.imwrite(str(out / 'tracking_online_lidar_last.png'), frames_img[-1])
    shutil.copy2(out / 'tracking_online_lidar_last.png', notes / 'tracking_online_lidar_last.png')
    print('WROTE', mp4, 'frames', len(frames_img), 'skipped', skipped)
    print('ONLINE_LONLY_DONE')


if __name__ == '__main__':
    main()
