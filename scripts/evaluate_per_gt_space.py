"""按 **GT 空间**（而不是系统空间）做多分类器对比（阶段 F 第二步）。

为什么需要这个
--------------
`evaluate_space_types.py` 是按**系统空间**统计的。实测发现不同分类器产出的系统空间
数并不相同（`TextMatching` 会把一个复合空间切成多块，`SAGEE` 是单标签、永不切分），
于是各分类器的分母不一样（39 张图上是 1404 vs 1614），严格说不可直接比。

本脚本改成以 **GT 空间为统计单位**：

1. 把每个系统空间按几何对齐到一个 GT 空间（与训练集导出同一套 IoU / 质心逻辑）；
2. 落到同一个 GT 空间下的系统空间**投票**汇总成一个预测（多数票；全弃权则弃权）；
3. 在这一个 GT 空间上判对/判错 —— 不管它是被 1 块还是 3 块系统空间覆盖。

这样分母恒等于「GT 空间数」，**切分与否都不影响可比性**。

组合（ensemble）
----------------
`--ensemble "A,B"` 表示：按 A、B 的顺序取第一个**不弃权**的预测。这正是系统的实际用法
（文本/LLM 先判，判不出来再交给 SAGE-E）。它会和单分类器一起进表。

用法::

    python scripts/evaluate_per_gt_space.py \
        --run SAGEE=output/holdout_thr000 \
        --run TextMatching=output/jsonld_rdp \
        --restrict-to output/holdout_thr000 \
        --label-space-from data/sagee_cad2graph_core.labels.json \
        --ensemble "TextMatching,SAGEE" \
        --out-json data/compare_per_gt_space.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import export_sagee_dataset as ex                            # noqa: E402
from src.spatial.sagee.graph import _as_polygon               # noqa: E402
from src.spatial.sagee.labels import LabelSpace, fold_label   # noqa: E402

ABSTAIN = '<未判定>'
OUTSIDE = '<空间外>'


def _spaces(path: str) -> list[tuple[object, list[str]]]:
    """取 (多边形, [bldg: 类型名])，忽略无几何的空间。"""
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
        out.append((poly, [t.split(':', 1)[1] for t in types if t.startswith('bldg:')]))
    return out


def _match_to_gt(sys_polys, gt_polys, min_iou: float) -> list[int]:
    """每个系统空间 -> GT 下标（-1 表示没配上）。逻辑同 ``export_sagee_dataset.match_labels``。"""
    out = [-1] * len(sys_polys)
    for i, poly in enumerate(sys_polys):
        if poly is None or poly.is_empty:
            continue
        best_j, best_iou = -1, 0.0
        for j, gpoly in enumerate(gt_polys):
            try:
                inter = poly.intersection(gpoly).area
                if inter <= 0:
                    continue
                iou = inter / (poly.area + gpoly.area - inter)
            except Exception:                                    # noqa: BLE001
                continue
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= min_iou:
            out[i] = best_j
            continue
        c = poly.centroid
        for j, gpoly in enumerate(gt_polys):
            try:
                if gpoly.contains(c):
                    out[i] = j
                    break
            except Exception:                                    # noqa: BLE001
                continue
    return out


def _vote(preds: list[str]) -> str:
    """多数票；全弃权 -> 弃权。并列时取票数最多的第一个（顺序稳定：按票数降序）。"""
    real = [p for p in preds if p != ABSTAIN]
    if not real:
        return ABSTAIN
    cnt = collections.Counter(real)
    top = max(cnt.values())
    return sorted(c for c, v in cnt.items() if v == top)[0]


def _metrics(cm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """cm: (n_class, n_class + 2)，最后两列是弃权 / 空间外。

    ⚠️ 弃权与空间外必须算漏检（FN）：不输出不等于判对。
    """
    block = cm[:, :cm.shape[0]]
    tp = np.diag(block).astype(np.float64)
    fp = block.sum(0).astype(np.float64) - tp
    fn = cm.sum(1).astype(np.float64) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    support = cm.sum(1).astype(np.float64)
    present = support > 0
    macro = float(f1[present].mean()) if present.any() else 0.0
    return macro, support, f1, present


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='按 GT 空间做多分类器对比')
    ap.add_argument('--run', action='append', required=True, metavar='NAME=DIR')
    ap.add_argument('--gt-dir', default=os.path.join(root, 'output', 'gt'))
    ap.add_argument('--label-space-from', required=True)
    ap.add_argument('--min-iou', type=float, default=0.3)
    ap.add_argument('--restrict-to', nargs='*', default=None)
    ap.add_argument('--ensemble', default=None,
                    help='逗号分隔的运行名，如 "TextMatching,SAGEE"：取第一个不弃权的结果')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    label_space = LabelSpace.load(args.label_space_from)
    rules = label_space.folding
    print('标签空间    : %s（%d 类）' % (label_space.space, label_space.n_class))
    print('类别        : %s' % list(label_space.classes))

    runs = []
    for spec in args.run:
        name, _, path = spec.partition('=')
        if os.path.isdir(path):
            runs.append((name, path))
        else:
            print('[!!] 目录不存在，跳过: %s' % path)
    if not runs:
        return 1
    names = [n for n, _ in runs]

    gt_files = {ex._key(p): p for p in sorted(glob.glob(os.path.join(args.gt_dir, '*.jsonld')))}
    per_run_files = []
    for _n, d in runs:
        per_run_files.append({ex._key(p): p
                              for p in sorted(glob.glob(os.path.join(d, '*.jsonld')))
                              if not p.endswith('_raw.jsonld')})

    allowed = None
    if args.restrict_to:
        allowed = set()
        for d in args.restrict_to:
            for p in glob.glob(os.path.join(d, '*.jsonld')):
                if not p.endswith('_raw.jsonld'):
                    allowed.add(ex._key(p))
        print('限定图纸集合: %d 张' % len(allowed))

    common = set(gt_files)
    for f in per_run_files:
        common &= set(f)
    common = sorted(common)
    if allowed is not None:
        common = [k for k in common if k in allowed]
    print('公共图纸数  : %d\n' % len(common))

    n_class = label_space.n_class
    cols = n_class + 2                       # + 弃权 + 空间外
    ci = label_space.index_of                 # 类名 -> 列下标

    def col_of(pred: str) -> int:
        """预测名 -> 混淆矩阵列下标（未判定 / 空间外各占最后一列）。"""
        if pred in ci:
            return ci[pred]
        return n_class if pred == ABSTAIN else n_class + 1

    cm = {n: np.zeros((n_class, cols), dtype=np.int64) for n in names}
    hit = {n: 0 for n in names}                 # 该分类器判对的 GT 空间集合大小
    both = only = 0                             # 两两互补性（仅 --run 两个时有意义）
    combo: collections.Counter = collections.Counter()
    ensemble_cm = np.zeros((n_class, cols), dtype=np.int64) if args.ensemble else None
    ens_names = [s.strip() for s in args.ensemble.split(',')] if args.ensemble else []

    for key in common:
        _ids, gt_polys, gt_raw = ex._gt_polygons(ex._load(gt_files[key]))
        gt_labels = [fold_label(x, rules) for x in gt_raw]
        gt_idx_of = {g: i for i, g in enumerate(gt_labels) if ci.get(g) is not None}
        if not gt_polys:
            continue

        voted: dict[str, dict[int, str]] = {}
        for name, files in zip(names, per_run_files):
            sys_spaces = _spaces(files[key])
            mapping = _match_to_gt([p for p, _ in sys_spaces], gt_polys, args.min_iou)
            buckets: dict[int, list[str]] = collections.defaultdict(list)
            for i, j in enumerate(mapping):
                if j < 0:
                    continue
                ls = sys_spaces[i][1]
                pred = fold_label(ls[0], rules) if ls else ABSTAIN
                if pred != ABSTAIN and ci.get(pred) is None:
                    pred = OUTSIDE
                buckets[j].append(pred)
            voted[name] = {j: _vote(ps) for j, ps in buckets.items()}

        for gi, g in enumerate(gt_labels):
            if g not in gt_idx_of:
                continue                                  # 该类不在本标签空间内
            for name in names:
                pred = voted[name].get(gi, ABSTAIN)
                cm[name][ci[g], col_of(pred)] += 1
                if pred == g:
                    hit[name] += 1
            if len(names) == 2:
                a = voted[names[0]].get(gi, ABSTAIN) == g
                b = voted[names[1]].get(gi, ABSTAIN) == g
                if a and b:
                    both += 1
                elif a:
                    combo['only_%s' % names[0]] += 1
                elif b:
                    combo['only_%s' % names[1]] += 1
                else:
                    combo['neither'] += 1
            if ensemble_cm is not None:
                pred = ABSTAIN
                for nm in ens_names:
                    if nm not in voted:
                        continue
                    p = voted[nm].get(gi, ABSTAIN)
                    if p == OUTSIDE:
                        continue
                    if p != ABSTAIN:
                        pred = p
                        break
                ensemble_cm[ci[g], col_of(pred)] += 1

    classes = list(label_space.classes)
    order = [(n, cm[n]) for n in names]
    if ensemble_cm is not None:
        order.append(('ENSEMBLE(%s)' % '>'.join(ens_names), ensemble_cm))

    print('=' * 78)
    print('%-24s %8s %8s %9s %9s %9s %9s'
          % ('分类器', 'GT空间', '准确率', '覆盖率', 'MacroF1', 'WeightedF1', '弃权数'))
    print('-' * 78)
    summary, per_class = {}, {}
    for name, m in order:
        macro, support, f1, present = _metrics(m)
        labeled = int(m[:, :n_class].sum())
        counted = int(m.sum())
        print('%-24s %8d %8.3f %9.3f %9.3f %9.3f %9d'
              % (name, counted, np.diag(m[:, :n_class]).sum() / max(1, counted),
                 labeled / max(1, counted), macro,
                 float((f1 * support).sum() / max(1, support.sum())),
                 counted - labeled))
        summary[name] = {'n_gt': counted, 'accuracy': float(np.diag(m[:, :n_class]).sum() / max(1, counted)),
                         'coverage': float(labeled / max(1, counted)),
                         'macro_f1': macro,
                         'weighted_f1': float((f1 * support).sum() / max(1, support.sum()))}
        per_class[name] = f1
    print('=' * 78)
    print('注：统计单位是 **GT 空间**；同一 GT 空间被多块系统空间覆盖时按多数票合并。')

    if len(names) == 2 and ens_names:
        n_gt = int(next(iter(cm.values())).sum())
        print('\n--- 互补性（共 %d 个 GT 空间）---' % n_gt)
        print('两分类器都判对      : %5d  (%.3f)' % (both, both / max(1, n_gt)))
        for nm in names:
            v = combo.get('only_%s' % nm, 0)
            print('仅 %-16s 判对: %5d  (%.3f)' % (nm, v, v / max(1, n_gt)))
        print('两者都判错          : %5d  (%.3f)' % (combo.get('neither', 0),
                                                     combo.get('neither', 0) / max(1, n_gt)))
        oracle = both + sum(combo.get('only_%s' % nm, 0) for nm in names)
        ens_hit = int(np.diag(ensemble_cm[:, :n_class]).sum())
        print('并集上界（任一判对） : %5d  (%.3f)  <- 组合器的天花板'
              % (oracle, oracle / max(1, n_gt)))
        print('规则组合器实际命中   : %5d  (%.3f)  <- "%s" 顺序取第一个不弃权'
              % (ens_hit, ens_hit / max(1, n_gt), '>'.join(ens_names)))

    print('\n--- 逐 GT 类别 F1 ---')
    hdr = '%-18s' % 'GT 类别' + ''.join('%16s' % n[:15] for n, _ in order)
    print(hdr)
    macro, support, _f1, present = _metrics(order[0][1])
    for i, cls in enumerate(classes):
        if not present[i]:
            continue
        print('%-18s' % cls + ''.join('%16.3f' % pc[i] for _n, pc in
                                      [(n, per_class[n]) for n, _ in order]))

    if args.out_json:
        payload = {'label_space': label_space.space, 'classes': classes,
                   'n_drawings': len(common), 'summary': summary,
                   'complementarity': {'both_correct': both,
                                       **{k: int(v) for k, v in combo.items()}},
                   'f1_by_class': {n: [float(x) for x in pc] for n, pc in per_class.items()},
                   'confusion': {n: m.tolist() for n, m in order},
                   'ensemble_order': ens_names}
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=float)
        print('\n结果已写出: %s' % args.out_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
