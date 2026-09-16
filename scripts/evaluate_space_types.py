"""多分类器「空间类型识别」端到端对比（阶段 F）。

口径
----
对每个分类器的**富化后** JSON-LD 产物目录：

1. 按几何 IoU 把系统空间对齐到 GT 空间（**与训练集导出完全相同的匹配逻辑**，
   复用 ``export_sagee_dataset.match_labels``，保证训练/评估同源）；
2. 把 GT 标签过一遍**与训练相同的折叠规则**（否则 CORE 空间的走廊合一会被误判为错）；
3. 只统计 GT 标签落在该标签空间内的行（被丢弃的类不该计入）；
4. 预测为空（置信度不足被丢弃 / 未判定）算**弃权**，单列一行。

指标
----
* ``accuracy``  = 判对 / 有 GT 可对的空间
* ``macro_f1``  = 只在 GT 里出现过的类别上取平均（长尾数据集必须看这个）
* ``coverage``  = 有输出的比例（弃权率 = 1 - coverage）

用法::

    python scripts/evaluate_space_types.py \
        --run SAGEE=output/jsonld_sagee \
        --run TextMatching=output/jsonld_rdp \
        --space CORE
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
from src.spatial.sagee.labels import LabelSpace, fold_label, folding_for  # noqa: E402

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


def _metrics(cm: np.ndarray, n_class: int) -> tuple[float, np.ndarray, np.ndarray]:
    """由（n_class × n_pred_cols）的混淆矩阵算逐类指标。

    ⚠️ **弃权必须算漏检（FN）**。分类器不输出时，本应识别的类别就是没被召回来；
    若把弃权行从分母里去掉，macro-F1 会被严重虚高（实测能从 0.46 抬到 0.72）。
    所以：``fn`` 取自**整行**（含弃权/空间外列），``fp`` 只取自真实类别列
    （弃权与空间外不是一个“被预测的类别”，不产生 FP）。
    """
    block = cm[:, :n_class]
    tp = np.diag(block).astype(np.float64)
    fp = block.sum(0).astype(np.float64) - tp
    fn = cm.sum(1).astype(np.float64) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    support = cm.sum(1).astype(np.float64)
    present = support > 0
    macro = float(f1[present].mean()) if present.any() else 0.0
    return macro, support, f1


def evaluate_run(sys_dir: str, gt_dir: str, space_name: str, label_space: LabelSpace,
                 min_iou: float, allowed_keys: set[str] | None = None) -> dict:
    rules = label_space.folding
    sys_files = {ex._key(p): p for p in sorted(glob.glob(os.path.join(sys_dir, '*.jsonld')))
                 if not p.endswith('_raw.jsonld')}
    gt_files = {ex._key(p): p for p in sorted(glob.glob(os.path.join(gt_dir, '*.jsonld')))}
    common = sorted(set(sys_files) & set(gt_files))
    if allowed_keys is not None:
        # ⚠️ 必须把**所有**分类器都限到同一批图纸上，否则分母不同、数字不可比
        common = [k for k in common if k in allowed_keys]

    pairs: collections.Counter = collections.Counter()
    n_sys = n_matched = n_counted = n_labeled = 0
    per_drawing = []

    for key in common:
        sys_spaces = _spaces(sys_files[key])
        gt_spaces = _spaces(gt_files[key])
        if not sys_spaces or not gt_spaces:
            continue
        gt_polys = [p for p, _ in gt_spaces]
        gt_labels = [fold_label(ls[0], rules) if ls else '' for _, ls in gt_spaces]

        labels, _how = ex.match_labels([p for p, _ in sys_spaces], list(range(len(gt_polys))),
                                       gt_polys, gt_labels, min_iou)
        correct_here = 0
        for i, gt_label in enumerate(labels):
            n_sys += 1
            if not gt_label:
                continue
            n_matched += 1
            if label_space.index_of.get(gt_label) is None:
                continue                      # 该类别不在本标签空间内，不计入
            n_counted += 1
            sys_labels = sys_spaces[i][1]
            pred = fold_label(sys_labels[0], rules) if sys_labels else ABSTAIN
            if pred != ABSTAIN and label_space.index_of.get(pred) is None:
                # 预测了一个不在本标签空间内的类别 —— 计入“空间外”，算错
                pred = OUTSIDE
            if pred != ABSTAIN:
                n_labeled += 1
            pairs[(gt_label, pred)] += 1
            if pred == gt_label:
                correct_here += 1
        per_drawing.append({'key': key, 'n_sys': len(sys_spaces),
                            'n_counted': sum(1 for x in labels if x), 'correct': correct_here})

    # ⚠️ 必须构造成**方阵**：GT 轴只有标签空间内的类，而预测轴还可能多出
    # 「未判定」与「空间外」。用 diag() 算 F1 的前提是方阵。
    classes = sorted({g for g, _ in pairs})
    preds = list(classes) + [ABSTAIN, OUTSIDE]
    ci = {c: i for i, c in enumerate(classes)}
    pi = {p: i for i, p in enumerate(preds)}
    cm = np.zeros((len(classes), len(preds)), dtype=np.int64)
    for (g, p), c in pairs.items():
        if g in ci and p in pi:
            cm[ci[g], pi[p]] += c

    correct = sum(c for (g, p), c in pairs.items() if g == p)
    macro, support, f1_by_class = _metrics(cm, len(classes))
    present = support > 0
    return {
        'run': os.path.basename(sys_dir.rstrip('/\\')),
        'n_drawings': len(common),
        'n_sys_spaces': n_sys,
        'n_matched': n_matched,
        'n_counted': n_counted,
        'n_labeled': n_labeled,
        'correct': correct,
        'accuracy': correct / max(1, n_counted),
        'coverage': n_labeled / max(1, n_counted),
        'macro_f1': macro,
        'weighted_f1': float((f1_by_class * support).sum() / max(1, support.sum())),
        'classes': classes, 'preds': preds, 'cm': cm,
        'f1_by_class': f1_by_class,
        'pairs': dict(pairs), 'per_drawing': per_drawing,
    }


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='多分类器空间类型对比')
    ap.add_argument('--run', action='append', required=True,
                    metavar='NAME=DIR', help='可重复；NAME 仅用于展示')
    ap.add_argument('--gt-dir', default=os.path.join(root, 'output', 'gt'))
    ap.add_argument('--space', default='CORE', choices=['FULL', 'CORE'])
    ap.add_argument('--label-space-from', default=None,
                    help='直接用一个 .labels.json（与权重同源，最严谨）；'
                         '默认优先用 SAGE-E 模型同目录的类别表，否则按 --space 推')
    ap.add_argument('--min-iou', type=float, default=0.3)
    ap.add_argument('--restrict-to', nargs='*', default=None,
                    help='限定在这些产物目录的图纸集合上评估（保证各分类器分母一致）')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    if args.label_space_from:
        label_space = LabelSpace.load(args.label_space_from)
    else:
        gt_labels = []
        for path in sorted(glob.glob(os.path.join(args.gt_dir, '*.jsonld'))):
            for _poly, ls in _spaces(path):
                if ls:
                    gt_labels.append(ls[0])
        label_space = LabelSpace.fit(gt_labels, space=args.space)
    print('标签空间    : %s（%d 类）' % (label_space.space, label_space.n_class))
    print('类别        : %s\n' % list(label_space.classes))

    allowed: set[str] | None = None
    if args.restrict_to:
        allowed = set()
        for d in args.restrict_to:
            for p in glob.glob(os.path.join(d, '*.jsonld')):
                if not p.endswith('_raw.jsonld'):
                    allowed.add(ex._key(p))
        print('限定图纸集合: %d 张（来自 %s）' % (len(allowed), ', '.join(args.restrict_to)))

    results = []
    for spec in args.run:
        name, _, path = spec.partition('=')
        if not os.path.isdir(path):
            print('[!!] 目录不存在，跳过: %s' % path)
            continue
        res = evaluate_run(path, args.gt_dir, label_space.space, label_space,
                           args.min_iou, allowed)
        res['name'] = name
        results.append(res)

    if not results:
        print('[!!] 没有可评估的运行')
        return 1

    print('=' * 78)
    print('%-16s %8s %8s %9s %9s %9s %9s'
          % ('分类器', '可判空间', '准确率', '覆盖率', 'MacroF1', 'WeightedF1', '弃权数'))
    print('-' * 78)
    for r in results:
        print('%-16s %8d %8.3f %9.3f %9.3f %9.3f %9d'
              % (r['name'], r['n_counted'], r['accuracy'], r['coverage'],
                 r['macro_f1'], r['weighted_f1'], r['n_counted'] - r['n_labeled']))
    print('=' * 78)
    print('注：可判空间 = 与 GT 匹配成功且 GT 类别在本标签空间内的系统空间数')

    for r in results:
        print('\n--- %s 逐 GT 类别 ---' % r['name'])
        print('%-18s %8s %8s %8s' % ('GT 类别', '样本', 'F1', '主要误判'))
        cm, preds = r['cm'], r['preds']
        for i, cls in enumerate(r['classes']):
            row = cm[i]
            f1 = r['f1_by_class'][i]
            others = sorted(((row[j], preds[j]) for j in range(len(preds)) if preds[j] != cls),
                            reverse=True)[:2]
            top = ', '.join('%s(%d)' % (p, c) for c, p in others if c)
            print('%-18s %8d %8.3f %8s' % (cls, int(row.sum()), f1, top))

    if args.out_json:
        payload = []
        for r in results:
            rec = {k: v for k, v in r.items()
                   if k not in ('cm', 'pairs', 'per_drawing', 'f1_by_class')}
            # np.float64 / np.int64 不是 JSON 原生类型，必须显式转
            rec['f1_by_class'] = [float(x) for x in r['f1_by_class']]
            rec['confusion'] = {'classes': r['classes'], 'preds': r['preds'],
                                'matrix': r['cm'].tolist()}
            rec['pairs'] = [{'gt': g, 'pred': p, 'n': int(c)}
                            for (g, p), c in sorted(r['pairs'].items(), key=lambda kv: -kv[1])]
            rec['per_drawing'] = r['per_drawing']
            payload.append(rec)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=float)
        print('\n结果已写出: %s' % args.out_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
