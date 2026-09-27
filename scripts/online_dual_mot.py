#!/usr/bin/env python3
"""
Online LiDAR+Camera 3D MOT (no radar, no BEVFusion).
- CenterPoint → lidar Update + birth
- StreamPETR 3D (global) → camera Update only (larger R)
Async: process earlier timestamp first within each sample pair
  (on nuScenes, cam/lidar share sample timestamp → lidar then camera).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

# reuse L-only tracker primitives
sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_lidar_mot import (  # noqa: E402
    Detection3D, SensorFrame, Track, Tracker, VEH, NUSC_CLASSES,
    load_centerpoint_map, frame_from_centerpoint, render_bev, pick_scene,
    K_GATE, K_LIDAR_POS_NOISE, K_LIDAR_YAW_NOISE, K_RELIABLE_RANGE, K_SMOOTH,
)

K_CAM_GATE = 5.0
K_CAM_POS_NOISE = 1.0  # larger than lidar 0.25
K_CAM_YAW_NOISE = 0.05


def update_camera(track: Track, det: Detection3D):
    z = det.xyz.astype(np.float64)
    scale = 1.0 if (det.distance_to_ego < 0 or det.distance_to_ego < K_RELIABLE_RANGE) else 2.0
    R = scale * K_CAM_POS_NOISE * np.eye(3)
    H = np.zeros((3, 9)); H[:, :3] = np.eye(3)
    track.motion_kf.update(z, R, H)
    track.position = track.motion_kf.state[:3].copy()
    track.velocity = track.motion_kf.state[3:6].copy()
    track.acceleration = track.motion_kf.state[6:9].copy()
    Ry = np.array([[scale * K_CAM_YAW_NOISE]])
    Hy = np.array([[1.0, 0.0]])
    track.yaw_kf.update(np.array([det.yaw]), Ry, Hy)
    track.yaw = float(track.yaw_kf.state[0])
    track.yaw_rate = float(track.yaw_kf.state[1])
    track.size = track.size * K_SMOOTH + det.size * (1 - K_SMOOTH)
    track.last_update_ts = det.timestamp
    # count toward confirm via lidar field reuse: half-credit as +1 lidar-equivalent
    track.associated_lidar += 1
    track.category = det.category


class DualTracker(Tracker):
    def run_lidar(self, frame: SensorFrame):
        return self.run(frame)

    def run_camera(self, frame: SensorFrame):
        """Associate + update only; NO birth from camera."""
        self.predict_all(frame.timestamp)
        dets = frame.detections
        if not dets or not self.tracks:
            return [t for t in self.tracks if t.is_confirmed()]

        used = [False] * len(self.tracks)
        for d in dets:
            best_j, best_dist = -1, K_CAM_GATE
            for j, tr in enumerate(self.tracks):
                if used[j]:
                    continue
                dist = float(np.linalg.norm(d.xyz - tr.position))
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j >= 0:
                used[best_j] = True
                update_camera(self.tracks[best_j], d)

        self.tracks = [t for t in self.tracks if not t.is_lost()]
        return [t for t in self.tracks if t.is_confirmed()]


def frame_from_streampetr(nusc, sample, anns, thr=0.35, maxn=40) -> SensorFrame:
    lid_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose', lid_sd['ego_pose_token'])
    ego_xy = np.array(pose['translation'][:2], dtype=np.float64)
    ego_yaw = Quaternion(pose['rotation']).yaw_pitch_roll[0]
    ts = sample['timestamp'] * 1e-6

    kept = [a for a in anns if a.get('detection_score', 0) >= thr and a.get('detection_name') in VEH]
    kept = sorted(kept, key=lambda a: -a['detection_score'])[:maxn]
    dets = []
    for k, a in enumerate(kept):
        box = Box(a['translation'], a['size'], Quaternion(a['rotation']),
                  name=a.get('detection_name', 'car'), score=float(a['detection_score']))
        xyz = np.array(box.center, dtype=np.float64)
        w, l, h = box.wlh
        size = np.array([l, w, h], dtype=np.float64)
        yaw = box.orientation.yaw_pitch_roll[0]
        dist = float(np.linalg.norm(xyz[:2] - ego_xy))
        dets.append(Detection3D(k, ts, xyz, size, yaw, float(a['detection_score']),
                                a.get('detection_name', 'car'), dist))
    return SensorFrame(0, ts, ego_xy, ego_yaw, dets, sample['token'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/cloud/cloud-ssd1/autonomous_driving_upgrade')
    ap.add_argument('--scene', default='scene-0103')
    ap.add_argument('--scenes', default='', help="comma list | 'val' | 'all'")
    ap.add_argument('--thr', type=float, default=0.35)
    ap.add_argument('--fps', type=float, default=5.0)
    ap.add_argument('--max-frames', type=int, default=0, help='per scene; 0 = all')
    ap.add_argument('--tag', default='long')
    args = ap.parse_args()

    root = Path(args.root)
    out = root / 'artifacts/day3_track/online'
    notes = root / 'notes/assets/tracking'
    out.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version='v1.0-mini', dataroot=str(root / 'data/nuscenes'), verbose=False)
    cp_map = load_centerpoint_map(root)
    sp_path = root / 'repos/StreamPETR/test/stream_petr_r50_flash_704_bs2_seq_90e/Sat_Sep_26_04_25_40_2026/pts_bbox/results_nusc.json'
    sp = json.load(open(sp_path))['results']

    from online_lidar_mot import resolve_scenes, scene_sample_chain
    scenes = resolve_scenes(nusc, args.scenes, args.scene)
    print('scenes', [s['name'] for s in scenes], 'max_frames', args.max_frames)

    Track._next_id = 1
    frames_img = []
    global_i = 0
    skipped = 0
    for scene in scenes:
        tracker = DualTracker()
        chain = scene_sample_chain(nusc, scene, args.max_frames)
        for i, sample in enumerate(chain):
            cp = cp_map.get(sample['token'])
            if cp is None:
                skipped += 1
                continue
            lidar_f = frame_from_centerpoint(nusc, sample, cp, thr=args.thr)
            lidar_f.index = global_i
            cam_f = frame_from_streampetr(nusc, sample, sp.get(sample['token'], []), thr=args.thr)
            cam_f.index = global_i

            pub = tracker.run_lidar(lidar_f)
            pub = tracker.run_camera(cam_f)

            title = (f'ONLINE L+C | CP+StreamPETR→峰华MOT | {scene["name"]} '
                     f'f={i+1}/{len(chain)} g={global_i+1} L={len(lidar_f.detections)} '
                     f'C={len(cam_f.detections)} tracks={len(pub)} | mini')
            frames_img.append(render_bev(lidar_f, pub, title))
            print(f'{scene["name"]} f{i}: L={len(lidar_f.detections)} C={len(cam_f.detections)} '
                  f'confirmed={len(pub)}')
            global_i += 1

    if not frames_img:
        raise SystemExit(f'no frames skipped={skipped}')

    tag = args.tag or 'long'
    mp4 = out / f'tracking_online_lidar_cam_{tag}.mp4'
    h, w = frames_img[0].shape[:2]
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    for im in frames_img:
        vw.write(im)
    vw.release()
    shutil.copy2(mp4, out / 'tracking_online_lidar_cam.mp4')
    shutil.copy2(mp4, notes / 'tracking_online_lidar_cam.mp4')
    shutil.copy2(mp4, notes / f'tracking_online_lidar_cam_{tag}.mp4')
    cv2.imwrite(str(out / 'tracking_online_lidar_cam_last.png'), frames_img[-1])
    shutil.copy2(out / 'tracking_online_lidar_cam_last.png', notes / 'tracking_online_lidar_cam_last.png')
    print('WROTE', mp4, 'frames', len(frames_img), 'skipped', skipped)
    print('ONLINE_LC_DONE')


if __name__ == '__main__':
    main()
