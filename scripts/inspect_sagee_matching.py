"""诊断：系统轮廓 ↔ GT 标注 的匹配质量。

导出脚本报出"标签分布和 GT 原始分布严重不符"时，用这个脚本看单张图纸的逐节点明细，
判断是**匹配错了**还是**系统根本没提取出那类空间**。

用法::

    python scripts/inspect_sagee_matching.py --key "2suite#1"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import export_sagee_dataset as ex  # noqa: E402  （复用配对与解析逻辑）


def _polys(labels_path: str) -> dict[str, Polygon]:
    out = {}
    for path in sorted(glob.glob(labels_path)):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for node in data.get('@graph') or []:
            if not isinstance(node, dict):
                continue
            types = node.get('@type') or []
            types = [types] if isinstance(types, str) else list(types)
            if 'bot:Space' not in types:
                continue
            wkt = node.get('geo:asWKT')
            if isinstance(wkt, dict):
                wkt = wkt.get('@value')
            if not wkt:
                continue
            try:
                poly = wkt_loads(wkt)
            except Exception:                                    # noqa: BLE001
                continue
            label = next((t.split(':', 1)[1] for t in types if t.startswith('bldg:')), None)
            if label:
                out[str(node.get('@id'))] = (poly, label)
    return out


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='检查系统轮廓与 GT 的匹配')
    ap.add_argument('--jsonld-dir', default=os.path.join(root, 'output', 'jsonld'))
    ap.add_argument('--gt-dir', default=os.path.join(root, 'output', 'gt'))
    ap.add_argument('--key', default=None, help='只查这一份（如 2suite#1）；默认查全部汇总')
    ap.add_argument('--min-iou', type=float, default=0.3)
    args = ap.parse_args()

    sys_files = {ex._key(p): p for p in sorted(glob.glob(os.path.join(args.jsonld_dir, '*_raw.jsonld')))}
    gt_files = {ex._key(p): p for p in sorted(glob.glob(os.path.join(args.gt_dir, '*.jsonld')))}
    common = sorted(set(sys_files) & set(gt_files))
    if args.key:
        common = [k for k in common if k == args.key]
        if not common:
            print('没有找到 %r，可用: %s' % (args.key, sorted(set(sys_files) & set(gt_files))[:10]))
            return 1

    gt_total: collections.Counter = collections.Counter()
    sys_total: collections.Counter = collections.Counter()
    matched_total: collections.Counter = collections.Counter()
    reuse: collections.Counter = collections.Counter()   # 一个 GT 被几个系统轮廓用了
    ratio_sum, ratio_n = 0.0, 0

    for key in common:
        with open(sys_files[key], encoding='utf-8') as f:
            graph_dict = json.load(f)
        gt_map = _polys(os.path.join(args.gt_dir, os.path.basename(gt_files[key])))
        for _pid, (_poly, label) in gt_map.items():
            gt_total[label] += 1

        sys_polys = ex._sys_polygons(graph_dict)
        gt_ids = list(gt_map)
        gt_polys = [gt_map[i][0] for i in gt_ids]
        gt_labels = [gt_map[i][1] for i in gt_ids]
        labels, how = ex.match_labels(sys_polys, gt_ids, gt_polys, gt_labels, args.min_iou)
        for lb in labels:
            if lb:
                matched_total[lb] += 1
        ratio_sum += (len(sys_polys) / max(1, len(gt_polys)))
        ratio_n += 1

        # 统计 GT 被复用的次数：系统轮廓数远少于 GT 时，多个系统房间会落到同一个 GT
        used = collections.Counter(lb for lb in labels if lb)
        for lb, c in used.items():
            if c > 1:
                reuse[lb] += c - 1

        if args.key:
            print('%s: 系统 %d 个轮廓, GT %d 个空间\n' % (key, len(sys_polys), len(gt_polys)))
            print('%-10s %-12s %-14s %10s %10s' % ('系统下标', '匹配类型', 'GT 标签', '系统面积', 'GT面积'))
            polys_gt = gt_polys
            for i, lb in enumerate(labels):
                poly = sys_polys[i]
                area = poly.area / 1e6 if poly is not None else 0.0
                if lb:
                    j = gt_labels.index(lb)
                    garea = polys_gt[j].area / 1e6
                else:
                    garea = 0.0
                print('%-10d %-12s %-14s %10.1f %10.1f'
                      % (i, ('-' if lb is None else 'ok'), (lb or '<未匹配>'), area, garea))

    print('\n=== 汇总 (%d 张图) ===' % len(common))
    print('系统轮廓数 / GT 空间数 的平均比值: %.2f' % (ratio_sum / max(1, ratio_n)))
    all_labels = sorted(set(gt_total) | set(matched_total))
    print('\n%-22s %10s %10s %10s %10s' % ('标签', 'GT总数', '匹配后', '覆盖率', '被复用'))
    for lb in all_labels:
        g, m = gt_total[lb], matched_total[lb]
        print('%-22s %10d %10d %9.0f%% %10d'
              % (lb, g, m, 100.0 * m / max(1, g), reuse[lb]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
