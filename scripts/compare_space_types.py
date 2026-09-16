"""把一次流水线产出的空间类型与 GT 逐空间对比（诊断用）。

按几何重叠把系统空间对齐到 GT 空间，打印「系统判定的类型 vs GT 类型」，并给出
混淆计数 —— 用来快速判断分类器的判定是否合理。

用法::

    python scripts/compare_space_types.py \
        --sys "output/_sagee_test/2suite (1).jsonld" \
        --gt  "output/gt/2suite_annotated (1)_gt.jsonld"
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spatial.sagee.graph import _as_polygon                # noqa: E402
from src.spatial.sagee.labels import fold_label, folding_for   # noqa: E402


def _spaces(path: str, prefix: str = 'bldg:') -> list[tuple[str, object, list[str]]]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    out = []
    for node in data.get('@graph') or []:
        if not isinstance(node, dict):
            continue
        types = node.get('@type') or []
        types = [types] if isinstance(types, str) else list(types)
        if 'bot:Space' not in types:
            continue
        poly = _as_polygon(node.get('geo:asWKT'))
        if poly is None:
            continue
        labels = [t.split(':', 1)[1] for t in types if t.startswith(prefix)]
        out.append((str(node.get('@id', '')), poly, labels))
    return out


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='系统空间类型 vs GT')
    ap.add_argument('--sys', default=os.path.join(root, 'output', '_sagee_test', '2suite (1).jsonld'))
    ap.add_argument('--gt', default=os.path.join(root, 'output', 'gt', '2suite_annotated (1)_gt.jsonld'))
    ap.add_argument('--space', default='CORE', choices=['FULL', 'CORE'],
                    help='GT 标签要用**与训练相同的折叠规则**再比，否则走廊合一会被误判为错')
    args = ap.parse_args()

    # ⚠️ 必须先把 GT 标签过一遍与训练一致的折叠规则：
    # CORE 空间里 MainCorridor/SecondaryCorridor/PublicCorridor 已统一成 Corridor，
    # 直接拿原始 GT 类型比会把正确的预测算成错误。
    rules = folding_for(args.space)

    def _fold(labels: list[str]) -> list[str]:
        return [fold_label(lb, rules) for lb in labels]

    sys_spaces = _spaces(args.sys)
    gt_spaces = [(i, p, _fold(l)) for i, p, l in _spaces(args.gt)]
    print('系统空间 %d 个 | GT 空间 %d 个（标签空间 %s）\n'
          % (len(sys_spaces), len(gt_spaces), args.space))

    agree = collections.Counter()
    total = 0
    rows = []
    for sid, poly, labels in sys_spaces:
        pred = labels[0] if labels else '<未判定>'
        best_j, best_iou = -1, 0.0
        for j, (_gid, gpoly, _gl) in enumerate(gt_spaces):
            try:
                inter = poly.intersection(gpoly).area
                if inter <= 0:
                    continue
                iou = inter / (poly.area + gpoly.area - inter)
            except Exception:                                    # noqa: BLE001
                continue
            if iou > best_iou:
                best_j, best_iou = j, iou
        gt_label = (gt_spaces[best_j][2][0] if best_j >= 0 and gt_spaces[best_j][2]
                    else '<无GT>')
        hit = 'OK ' if pred == gt_label else '   '
        rows.append((hit, sid, pred, gt_label, poly.area / 1e6, best_iou))
        total += 1
        agree[(gt_label, pred)] += 1

    print('%-6s %-14s %-16s %-16s %8s %7s' % ('', '空间', '系统判定', 'GT', '面积m²', 'IoU'))
    for hit, sid, pred, gt_label, area, iou in rows:
        print('%-6s %-14s %-16s %-16s %8.1f %7.2f'
              % (hit, sid.replace('inst:', ''), pred, gt_label, area, iou))

    correct = sum(1 for r in rows if r[0] == 'OK ')
    print('\n逐空间完全一致: %d / %d = %.1f%%' % (correct, total, 100.0 * correct / max(1, total)))

    print('\n混淆（行=GT，列=系统判定）:')
    for (g, p), c in sorted(agree.items(), key=lambda kv: -kv[1]):
        print('  %-18s -> %-18s %d' % (g, p, c))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
