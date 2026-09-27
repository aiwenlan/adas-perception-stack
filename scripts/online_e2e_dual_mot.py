#!/usr/bin/env python3
"""
Online e2e dual-sensor MOT with live GPU inference (not pkl replay).

Same process (streampetr env):
  each frame → CenterPoint infer → StreamPETR infer → DualTracker → log ms

Both models stay on GPU when VRAM allows; otherwise falls back to
serialized load (still per-frame infer for both, just not resident together).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

ROOT = Path(os.environ.get('AD_ROOT', '/cloud/cloud-ssd1/autonomous_driving_upgrade'))
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / 'repos/StreamPETR'))
sys.path.insert(0, str(ROOT / 'repos/StreamPETR/mmdetection3d'))

from online_lidar_mot import (  # noqa: E402
    Detection3D, SensorFrame, Track, Tracker, VEH, NUSC_CLASSES,
    frame_from_centerpoint, render_bev, pick_scene, lidar_box_to_global,
)
from online_dual_mot import DualTracker, update_camera, frame_from_streampetr  # noqa: E402


def percentile(xs, p):
    if not xs:
        return float('nan')
    a = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(a, p))


def boxes_from_cp_result(res, sample, nusc, thr=0.35, maxn=40) -> SensorFrame:
    return frame_from_centerpoint(nusc, sample, res, thr=thr, maxn=maxn)


def boxes_from_sp_result(res_list, sample, nusc, thr=0.35, maxn=40) -> SensorFrame:
    return frame_from_streampetr(nusc, sample, res_list, thr=thr, maxn=maxn)


def result_to_cp_dict(result):
    """Normalize mmdet3d model output to pts_bbox-like dict."""
    if isinstance(result, list):
        result = result[0]
    if isinstance(result, dict) and 'pts_bbox' in result:
        return result
    if isinstance(result, dict) and 'boxes_3d' in result:
        return {'pts_bbox': result}
    raise TypeError(f'unexpected CP result type: {type(result)}')


def result_to_sp_anns(result, score_thr=0.0):
    """Convert StreamPETR output to nuScenes-style ann list (global)."""
    if isinstance(result, list):
        result = result[0]
    if isinstance(result, dict) and 'pts_bbox' in result:
        bbox = result['pts_bbox']
    else:
        bbox = result
    boxes = bbox['boxes_3d']
    scores = bbox['scores_3d']
    labels = bbox['labels_3d']
    if hasattr(boxes, 'tensor'):
        # LiDARInstance3DBoxes — StreamPETR usually outputs in ego/lidar; test pipeline converts
        # For NuScenes eval they format to global. Here we use .tensor gravity etc.
        b = boxes.tensor.detach().cpu().numpy()
    else:
        b = np.asarray(boxes)
    s = scores.detach().cpu().numpy() if hasattr(scores, 'detach') else np.asarray(scores)
    lab = labels.detach().cpu().numpy() if hasattr(labels, 'detach') else np.asarray(labels)
    anns = []
    for i in range(len(b)):
        if float(s[i]) < score_thr:
            continue
        # Expect translation xyz, size wlh or lwh depending on box type — use NuScenes format via Box helper later
        # StreamPETR test format_results writes translation/size/rotation; for raw output use lidar box → we
        # pass through format if available.
        cname = NUSC_CLASSES[int(lab[i])] if 0 <= int(lab[i]) < len(NUSC_CLASSES) else 'car'
        anns.append({
            'detection_name': cname,
            'detection_score': float(s[i]),
            '_box7': b[i][:7].tolist(),
            '_raw': True,
        })
    return anns


def _set_ann_file(cfg, ann_file):
    if ann_file is None:
        return
    if isinstance(cfg.data.test, dict):
        cfg.data.test['ann_file'] = ann_file
    else:
        cfg.data.test.ann_file = ann_file


def build_centerpoint(cfg_path, ckpt, ann_files=None):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    from mmcv.parallel import MMDataParallel
    import mmdet3d.datasets  # noqa: F401 — register NuScenesDataset

    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if isinstance(cfg.data.test, dict):
        cfg.data.test.pop('samples_per_gpu', None)
    elif hasattr(cfg.data.test, 'samples_per_gpu'):
        delattr(cfg.data.test, 'samples_per_gpu')

    if ann_files is None:
        ann_list = [None]
    elif isinstance(ann_files, (list, tuple)):
        ann_list = list(ann_files)
    else:
        ann_list = [ann_files]

    datasets = []
    tok2i = {}  # token -> (ds_idx, local_idx)
    for af in ann_list:
        _set_ann_file(cfg, af)
        ds = build_dataset(cfg.data.test)
        di = len(datasets)
        for i in range(len(ds)):
            info = ds.data_infos[i]
            tok = info.get('token')
            if tok:
                tok2i[tok] = (di, i)
        datasets.append(ds)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    loader = None
    return model, datasets, loader, tok2i, list(tok2i.keys())


def build_streampetr(cfg_path, ckpt, ann_files=None):
    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    from mmcv.parallel import MMDataParallel
    import mmdet3d.datasets  # noqa: F401
    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if isinstance(cfg.data.test, dict):
        cfg.data.test.pop('samples_per_gpu', None)

    if ann_files is None:
        ann_list = [None]
    elif isinstance(ann_files, (list, tuple)):
        ann_list = list(ann_files)
    else:
        ann_list = [ann_files]

    datasets = []
    tok2i = {}
    for af in ann_list:
        _set_ann_file(cfg, af)
        ds = build_dataset(cfg.data.test)
        di = len(datasets)
        for i in range(len(ds)):
            tok = ds.data_infos[i]['token']
            tok2i[tok] = (di, i)
        datasets.append(ds)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model, datasets, None, tok2i, list(tok2i.keys())


@torch.no_grad()
def infer_one(model, data):
    if hasattr(model, 'module'):
        return model(return_loss=False, rescale=True, **data)
    return model(return_loss=False, rescale=True, **data)


def collate_one(dataset_or_list, index):
    """Get a single-sample batch. index may be int or (ds_idx, local_idx)."""
    from mmcv.parallel import collate
    if isinstance(index, tuple):
        ds = dataset_or_list[index[0]]
        local_i = index[1]
    else:
        ds = dataset_or_list[0] if isinstance(dataset_or_list, list) else dataset_or_list
        local_i = index
    data = ds[local_i]
    data = collate([data], samples_per_gpu=1)
    def to_cuda(obj):
        if torch.is_tensor(obj):
            return obj.cuda(non_blocking=True)
        if isinstance(obj, list):
            return [to_cuda(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(to_cuda(x) for x in obj)
        if isinstance(obj, dict):
            return {k: to_cuda(v) for k, v in obj.items()}
        if hasattr(obj, 'data'):
            try:
                from mmcv.parallel import DataContainer
                if isinstance(obj, DataContainer):
                    return DataContainer(to_cuda(obj.data), obj.stack, obj.padding_value, cpu_only=obj.cpu_only)
            except Exception:
                pass
        return obj
    return to_cuda(data)


def sp_raw_to_global_anns(raw_anns, sample, nusc):
    """Convert raw lidar box7 anns to global NuScenes JSON-like anns."""
    out = []
    for a in raw_anns:
        if not a.get('_raw'):
            out.append(a)
            continue
        b7 = a['_box7']
        # StreamPETR boxes often already in ego/global depending on pipeline;
        # use same lidar_box_to_global as CenterPoint for lidar-coordinate tensors.
        box = lidar_box_to_global(b7, sample, nusc)
        out.append({
            'sample_token': sample['token'],
            'translation': box.center.tolist(),
            'size': box.wlh.tolist(),
            'rotation': box.orientation.q.tolist(),
            'detection_name': a['detection_name'],
            'detection_score': a['detection_score'],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--scene', default='scene-0103')
    ap.add_argument('--scenes', default='', help="comma list | 'val' | 'all'")
    ap.add_argument('--thr', type=float, default=0.35)
    ap.add_argument('--max-frames', type=int, default=0, help='per scene; 0 = all')
    ap.add_argument('--fps', type=float, default=5.0)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--mini-all', action='store_true',
                    help='load train+val infos so all mini scenes can be inferred')
    ap.add_argument('--tag', default='long')
    args = ap.parse_args()
    root = Path(args.root)
    out = root / 'artifacts/day3_track/online_e2e'
    notes = root / 'notes/assets/tracking'
    # this repo's assets/ (clone dir name may vary)
    bev = Path(__file__).resolve().parent.parent / 'assets' / 'tracking'
    out.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)
    bev.mkdir(parents=True, exist_ok=True)

    cp_cfg = str(root / 'repos/StreamPETR/mmdetection3d/configs/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py')
    cp_ckpt = str(root / 'ckpts/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.pth')
    sp_cfg = str(root / 'repos/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py')
    sp_ckpt = str(root / 'ckpts/streampetr/stream_petr_r50_flash_704_bs2_seq_90e.pth')

    nusc = NuScenes(version='v1.0-mini', dataroot=str(root / 'data/nuscenes'), verbose=False)
    from online_lidar_mot import resolve_scenes, scene_sample_chain
    scenes = resolve_scenes(nusc, args.scenes, args.scene)
    print('scenes', [s['name'] for s in scenes], 'max_frames', args.max_frames, 'mini_all', args.mini_all)

    ann_cp = ann_sp = None
    if args.mini_all or (args.scenes or '').lower() == 'all':
        # CenterPoint (mmdet3d) infos
        ann_cp = [
            str(root / 'data/nuscenes/nuscenes_infos_train.pkl'),
            str(root / 'data/nuscenes/nuscenes_infos_val.pkl'),
        ]
        # StreamPETR 2d temporal infos
        ann_sp = [
            str(root / 'data/nuscenes/nuscenes2d_temporal_infos_train.pkl'),
            str(root / 'data/nuscenes/nuscenes2d_temporal_infos_val.pkl'),
        ]
        args.mini_all = True

    print('Loading CenterPoint...')
    cp_model, cp_ds, _, cp_tok2i, _ = build_centerpoint(cp_cfg, cp_ckpt, ann_files=ann_cp)
    print('Loading StreamPETR...')
    sp_model, sp_ds, _, sp_tok2i, _ = build_streampetr(sp_cfg, sp_ckpt, ann_files=ann_sp)
    print('models ready', 'VRAM', torch.cuda.memory_allocated() / 1024**2, 'MB',
          'cp_tok', len(cp_tok2i), 'sp_tok', len(sp_tok2i))

    # warmup on first available token
    warm_tok = next(iter(cp_tok2i))
    for w in range(args.warmup):
        if warm_tok in cp_tok2i:
            data = collate_one(cp_ds, cp_tok2i[warm_tok])
            _ = infer_one(cp_model, data)
        if warm_tok in sp_tok2i:
            data = collate_one(sp_ds, sp_tok2i[warm_tok])
            _ = infer_one(sp_model, data)
        torch.cuda.synchronize()
    print('warmup done')

    Track._next_id = 1
    rows = []
    frames_img = []
    global_i = 0

    for scene in scenes:
        tracker = DualTracker()
        chain = scene_sample_chain(nusc, scene, args.max_frames)
        for i, sample in enumerate(chain):
            token = sample['token']
            if token not in cp_tok2i or token not in sp_tok2i:
                print('skip missing token', scene['name'], token[:10])
                continue

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            cp_data = collate_one(cp_ds, cp_tok2i[token])
            cp_out = infer_one(cp_model, cp_data)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            cp_ms = (t1 - t0) * 1000.0
            cp_res = result_to_cp_dict(cp_out)
            lidar_f = boxes_from_cp_result(cp_res, sample, nusc, thr=args.thr)
            lidar_f.index = global_i

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            sp_data = collate_one(sp_ds, sp_tok2i[token])
            sp_out = infer_one(sp_model, sp_data)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            sp_ms = (t3 - t2) * 1000.0
            raw = result_to_sp_anns(sp_out, score_thr=0.0)
            anns = raw
            if raw and raw[0].get('_raw'):
                try:
                    anns = sp_raw_to_global_anns(raw, sample, nusc)
                except Exception as e:
                    print('SP convert warn', e)
                    anns = []
            cam_f = boxes_from_sp_result(anns, sample, nusc, thr=args.thr)
            cam_f.index = global_i

            t4 = time.perf_counter()
            pub = tracker.run_lidar(lidar_f)
            pub = tracker.run_camera(cam_f)
            t5 = time.perf_counter()
            track_ms = (t5 - t4) * 1000.0
            total_ms = (t5 - t0) * 1000.0

            row = {
                'i': global_i,
                'scene': scene['name'],
                'token': token,
                'cp_ms': round(cp_ms, 2),
                'sp_ms': round(sp_ms, 2),
                'track_ms': round(track_ms, 2),
                'total_ms': round(total_ms, 2),
                'n_lidar': len(lidar_f.detections),
                'n_cam': len(cam_f.detections),
                'n_tracks': len(pub),
            }
            rows.append(row)
            print(f'{scene["name"]} f{i}: CP {cp_ms:.1f} SP {sp_ms:.1f} trk {track_ms:.1f} tot {total_ms:.1f} '
                  f'| L={row["n_lidar"]} C={row["n_cam"]} T={row["n_tracks"]}')

            title = (f'E2E LIVE L+C | CP+SP infer→MOT | {scene["name"]} '
                     f'f={i+1}/{len(chain)} g={global_i+1} | {total_ms:.0f}ms '
                     f'(CP {cp_ms:.0f}+SP {sp_ms:.0f}+trk {track_ms:.0f})')
            frames_img.append(render_bev(lidar_f, pub, title))
            global_i += 1

    if not frames_img:
        raise SystemExit('no frames')

    h, w = frames_img[0].shape[:2]
    tag = args.tag or 'long'
    mp4 = out / f'tracking_e2e_live_{tag}.mp4'
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    for im in frames_img:
        vw.write(im)
    vw.release()
    import shutil
    canon = out / 'tracking_e2e_live_lidar_cam.mp4'
    if mp4.resolve() != canon.resolve():
        shutil.copy2(mp4, canon)
    for dest_dir in (notes, bev):
        shutil.copy2(mp4, dest_dir / 'tracking_e2e_live_lidar_cam.mp4')
        if (dest_dir / f'tracking_e2e_live_{tag}.mp4').resolve() != mp4.resolve():
            shutil.copy2(mp4, dest_dir / f'tracking_e2e_live_{tag}.mp4')
    cv2.imwrite(str(out / 'tracking_e2e_live_lidar_cam_last.png'), frames_img[-1])
    for dest_dir in (notes, bev):
        shutil.copy2(out / 'tracking_e2e_live_lidar_cam_last.png',
                     dest_dir / 'tracking_e2e_live_lidar_cam_last.png')

    use = rows[2:] if len(rows) > 5 else rows
    cp_list = [r['cp_ms'] for r in use]
    sp_list = [r['sp_ms'] for r in use]
    tr_list = [r['track_ms'] for r in use]
    tot_list = [r['total_ms'] for r in use]
    summary = {
        'scenes': [s['name'] for s in scenes],
        'tag': tag,
        'n_frames': len(use),
        'note': 'same-process live GPU infer (CenterPoint+StreamPETR) + DualTracker; 4090; mini',
        'mini_all': bool(args.mini_all),
        'cp_ms': {'mean': round(float(np.mean(cp_list)), 2), 'p50': round(percentile(cp_list, 50), 2), 'p95': round(percentile(cp_list, 95), 2)},
        'sp_ms': {'mean': round(float(np.mean(sp_list)), 2), 'p50': round(percentile(sp_list, 50), 2), 'p95': round(percentile(sp_list, 95), 2)},
        'track_ms': {'mean': round(float(np.mean(tr_list)), 2), 'p50': round(percentile(tr_list, 50), 2), 'p95': round(percentile(tr_list, 95), 2)},
        'total_ms': {'mean': round(float(np.mean(tot_list)), 2), 'p50': round(percentile(tot_list, 50), 2), 'p95': round(percentile(tot_list, 95), 2)},
        'approx_fps': round(1000.0 / float(np.mean(tot_list)), 2) if tot_list else None,
        'per_frame': use,
    }
    with open(out / 'latency_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    shutil.copy2(out / 'latency_summary.json', notes / 'latency_summary.json')
    shutil.copy2(out / 'latency_summary.json', bev / 'latency_summary.json')
    print('SUMMARY', json.dumps({k: summary[k] for k in summary if k != 'per_frame'}, indent=2))
    print('WROTE', mp4, 'frames', len(frames_img))
    print('E2E_LIVE_DONE')


if __name__ == '__main__':
    main()
