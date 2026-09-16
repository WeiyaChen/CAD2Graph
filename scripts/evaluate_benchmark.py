"""评估 ``output/bench/`` 下的基准矩阵，产出网页端要读的 ``scores.json``。

两个任务分别汇总
----------------
* **任务 1 · 空间轮廓提取** —— 对每个轮廓方法算 :func:`contour_metrics`
  （覆盖率 / 一对一率 / mIoU / 面积误差 / 数量比）。
* **任务 2 · 空间类型识别** —— 对每个分类器算空间类型指标
  （accuracy / coverage / macro-F1 / weighted-F1），**以 GT 空间为统计单位**。

指标实现全部来自 :mod:`src.experiment.bench_metrics` —— 与网页端、与
``scripts/evaluate_per_gt_space.py`` 共用一份，避免"同一口径两处实现"。

``sample`` 的处理
-----------------
用户明确要求 benchmark 不计入 ``sample``（它只是跑通流程的样例）。批处理无法按文件名
过滤，所以这里统一排除键以 ``sample`` 开头的图纸。

用法::

    python scripts/evaluate_benchmark.py
    python scripts/evaluate_benchmark.py --out-json output/bench/scores.json
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiment import bench_metrics as bm                      # noqa: E402
from src.spatial.sagee.labels import LabelSpace, fold_label          # noqa: E402

#: 目录名前缀 -> 它属于哪个任务
CONTOUR_PREFIX = 'contour_'
TYPE_PREFIX = 'type_'
E2E_PREFIX = 'e2e_'


def discover(bench_dir: str) -> dict[str, list[dict]]:
    """按**目录名前缀**归类矩阵里的 run。

    刻意从目录名推断而不是读 manifest：目录名是唯一真相，manifest 可能是旧配方留下的。
    命名约定：

        ``contour_<轮廓算法>``      任务 1 的 run（分类器固定 ``NoOp``）
        ``type_<分类器>``           任务 2 的 run（轮廓固定 ``GT``）
        ``e2e_<轮廓算法>_<分类器>`` 端到端，任务 2 跑在任务 1 的轮廓上
    """
    out: dict[str, list[dict]] = {'contour': [], 'type': [], 'e2e': []}
    for d in sorted(glob.glob(os.path.join(bench_dir, '*'))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if name.startswith(E2E_PREFIX):
            # 两种算法名里都不带下划线，所以 split 一次就能切开
            contour, _, clf = name[len(E2E_PREFIX):].partition('_')
            out['e2e'].append({'name': name, 'dir': d, 'contour': contour,
                               'clf': clf, 'method': '%s|%s' % (contour, clf)})
            continue
        for kind, prefix in (('contour', CONTOUR_PREFIX), ('type', TYPE_PREFIX)):
            if name.startswith(prefix):
                out[kind].append({'name': name, 'dir': d,
                                  'method': name[len(prefix):]})
                break
    return out


def e2e_breakdown(res: dict) -> dict:
    """把端到端准确率拆成「轮廓先找到了这个空间吗」×「找到之后类型判对了吗」。

    ⚠️ **不能**从混淆矩阵的最后一列 ``<空间外>`` 去反推“没覆盖”。
    ``match_to_gt`` 的方向是**系统空间 → 一个 GT 空间**，所以：

    * ``<空间外>``   = 某个系统空间被预测成了**标签空间之外**的类型；
    * ``<未判定>``   = 分类器弃权 **或** 没有任何系统空间落到这个 GT 上 —— 两者混在一起。

    因此覆盖率必须由 ``evaluate_type`` 直接统计（它手上有 ``buckets``），
    它随结果一起返回 ``n_covered_gt``。

    三个数的关系（自洽）：

        端到端准确率 = 轮廓覆盖 × 条件准确率          （= 判对数 / GT 空间总数）
    """
    a = res['aggregate']
    n_gt = int(a['n_gt'])
    covered = int(res.get('n_covered_gt', 0))
    # 没被覆盖的 GT 空间绝不可能判对，所以判对数只会落在已覆盖那部分里
    correct = round(a['accuracy'] * n_gt) if n_gt else 0
    return {
        'n_gt': n_gt,
        'n_covered': covered,
        'n_uncovered': n_gt - covered,
        'n_correct': int(correct),
        'contour_coverage': (covered / n_gt) if n_gt else 0.0,
        'accuracy_given_covered': (correct / covered) if covered else 0.0,
        'accuracy': (correct / n_gt) if n_gt else 0.0,
    }


def _keys_in(out_dir: str, gt_dir: str, exclude_prefixes: tuple[str, ...]) -> list[str]:
    """该产物目录 ∩ GT 目录 的图纸键（排除 sample 之类）。"""
    sys_keys = {bm.drawing_key(p) for p in glob.glob(os.path.join(out_dir, '*.jsonld'))
                if not p.endswith('_raw.jsonld')}
    gt_keys = {bm.drawing_key(p) for p in glob.glob(os.path.join(gt_dir, '*.jsonld'))}
    keys = sorted(sys_keys & gt_keys)
    return [k for k in keys
            if not any(k.lower().startswith(p.lower()) for p in exclude_prefixes)]


def _gt_path(gt_dir: str, key: str, cache: dict) -> str | None:
    if key in cache:
        return cache[key]
    for p in glob.glob(os.path.join(gt_dir, '*.jsonld')):
        if bm.drawing_key(p) == key:
            cache[key] = p
            return p
    cache[key] = None
    return None


def _sys_path(out_dir: str, key: str, cache: dict, raw: bool = False) -> str | None:
    """找该图纸的系统产物；``raw=True`` 取 ``<base>_raw.jsonld``。

    ⚠️ **任务 1 必须用 raw。** ``_raw.jsonld`` 是 ``topology_builder.build()`` 在富化
    **之前**写出的图，只有 ``bot:Space`` 和几何，在任何分类器下都逐位相同 —— 它就是
    "提取出来的轮廓"本身。富化后的产物可能已被 **ACD 复合空间切分**改动：一个空间若被
    判了多个类型会被切开（实测同一张图在 TextMatching 下 42 个空间、单标签分类器下 36 个）。
    拿富化后的产物去量轮廓质量，量到的是"分类器 + 提取器"的混合物。
    """
    ck = (key, raw)
    if ck in cache:
        return cache[ck]
    for p in glob.glob(os.path.join(out_dir, '*.jsonld')):
        if bm.drawing_key(p) != key:
            continue
        if raw != p.endswith('_raw.jsonld'):
            continue
        cache[ck] = p
        return p
    cache[ck] = None
    return None


# --------------------------------------------------------------------------- #
# 任务 1：轮廓
# --------------------------------------------------------------------------- #
def _pool_contour(rows: list[dict]) -> dict:
    """把逐图指标汇总成总量指标（按计数加权，不是简单平均）。"""
    agg = {k: 0.0 for k in ('n_sys', 'n_gt', 'n_covered_gt', 'n_matched')}
    for r in rows:
        for k in agg:
            agg[k] += r[k]
    w_miou = sum(r['miou_matched'] * r['n_matched'] for r in rows)
    w_area = sum(r['area_mae_m2'] * r['n_matched'] for r in rows)
    n_matched = agg['n_matched'] or 1
    agg.update({
        'count_ratio': agg['n_sys'] / agg['n_gt'] if agg['n_gt'] else 0.0,
        'coverage': agg['n_covered_gt'] / agg['n_gt'] if agg['n_gt'] else 0.0,
        # one_to_one 不是可加的，用逐图比例按 GT 数加权
        'one_to_one': (sum(r['one_to_one'] * r['n_gt'] for r in rows)
                       / agg['n_gt']) if agg['n_gt'] else 0.0,
        'miou_matched': w_miou / n_matched,
        'area_mae_m2': w_area / n_matched,
    })
    for k in ('n_sys', 'n_gt', 'n_covered_gt', 'n_matched'):
        agg[k] = int(agg[k])
    return agg


def evaluate_contour(run: dict, gt_dir: str, keys: list[str], min_iou: float) -> dict:
    gt_cache: dict = {}
    sys_cache: dict = {}
    per_drawing, rows = {}, []
    for key in keys:
        gp = _gt_path(gt_dir, key, gt_cache)
        sp = _sys_path(run['dir'], key, sys_cache, raw=True)   # ⚠️ 任务 1 读富化前的图
        if not gp or not sp:
            continue
        _ids, gt_polys, _labels = bm.load_gt(gp)
        sys_spaces = bm.load_system_spaces(sp)
        m = bm.contour_metrics([p for p, _ in sys_spaces], gt_polys, min_iou)
        per_drawing[key] = m
        rows.append(m)
    return {'aggregate': _pool_contour(rows), 'per_drawing': per_drawing,
            'n_drawings': len(rows)}


# --------------------------------------------------------------------------- #
# 任务 2：空间类型
# --------------------------------------------------------------------------- #
def evaluate_type(run: dict, gt_dir: str, keys: list[str], label_space: LabelSpace,
                  min_iou: float) -> dict:
    rules = label_space.folding
    index_of = label_space.index_of
    classes = list(label_space.classes)
    gt_cache: dict = {}
    per_drawing = {}
    cm_total = np.zeros((len(classes), len(classes) + 2), dtype=np.int64)
    n_covered_gt = 0          # 有至少一个系统空间对齐上来的 GT 空间数（供端到端分解用）
    for key in keys:
        gp = _gt_path(gt_dir, key, gt_cache)
        sp = _sys_path(run['dir'], key, {})          # 任务 2 读富化后的图（带类型）
        if not gp or not sp:
            continue
        _ids, gt_polys, gt_raw = bm.load_gt(gp)
        gt_folded = [fold_label(x, rules) for x in gt_raw]
        sys_spaces = bm.load_system_spaces(sp)
        mapping = bm.match_to_gt([p for p, _ in sys_spaces], gt_polys, min_iou)

        buckets: dict[int, list[str]] = collections.defaultdict(list)
        for i, j in enumerate(mapping):
            if j < 0:
                continue
            labels = sys_spaces[i][1]
            pred = fold_label(labels[0], rules) if labels else bm.ABSTAIN
            if pred != bm.ABSTAIN and index_of.get(pred) is None:
                pred = bm.OUTSIDE
            buckets[j].append(pred)

        pairs: collections.Counter = collections.Counter()
        for gi, g in enumerate(gt_folded):
            if index_of.get(g) is None:
                continue                              # 该类不在本标签空间内
            if buckets.get(gi):
                n_covered_gt += 1
            pairs[(g, bm.vote(buckets.get(gi, [bm.ABSTAIN])))] += 1
        cm = bm.type_confusion(pairs, classes, index_of)
        cm_total += cm
        per_drawing[key] = bm.metrics_from_cm(cm)

    return {'aggregate': bm.metrics_from_cm(cm_total),
            'per_drawing': per_drawing,
            'confusion_matrix': cm_total.tolist(),
            'n_covered_gt': n_covered_gt,
            'n_drawings': len(per_drawing)}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='评估基准矩阵')
    ap.add_argument('--bench-dir', default=os.path.join(root, 'output', 'bench'))
    ap.add_argument('--gt-dir', default=os.path.join(root, 'output', 'gt'))
    ap.add_argument('--label-space-from', default=os.path.join(
        root, 'data', 'cad2graph_sagee_gtc.labels.json'),
        help='任务 2 的标签空间。默认用**基于 GT 轮廓**导出的那份（CORE 16 类）：'
             '任务 2 的起点就是 GT 轮廓，标签空间理应按该起点能达成的类别来定。'
             '用 RGP 轮廓导出的那份只有 14 类 —— 因为 RGP 会把 ElectricalRoom / '
             'WaterRoom 合并丢掉；而在 GT 轮廓上这两类是真实存在的，'
             '若把它们排除在标签空间外，预测正确的空间反而会被判成「空间外」。')
    ap.add_argument('--min-iou', type=float, default=0.3)
    ap.add_argument('--exclude', nargs='*', default=['sample'],
                    help='排除的图纸键前缀（默认 sample）')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    found = discover(args.bench_dir)
    contour_runs, type_runs, e2e_runs = found['contour'], found['type'], found['e2e']
    if not contour_runs and not type_runs and not e2e_runs:
        print('[!!] %s 下没有 contour_* / type_* / e2e_* 目录' % args.bench_dir)
        return 1
    label_space = LabelSpace.load(args.label_space_from)

    # 把 GT 文件路径写进产物，供网页端直接取用（否则前端只能猜文件名，很容易猜错）
    gt_paths: dict[str, str] = {}
    for p in glob.glob(os.path.join(args.gt_dir, '*.jsonld')):
        k = bm.drawing_key(p)
        if any(k.lower().startswith(x.lower()) for x in (args.exclude or [])):
            continue
        gt_paths[k] = os.path.relpath(p, root).replace('\\', '/')

    print('基准目录    : %s' % os.path.relpath(args.bench_dir, root))
    print('标签空间    : %s（%d 类）%s'
          % (label_space.space, label_space.n_class, list(label_space.classes)))
    print('排除图纸前缀: %s\n' % (args.exclude or '（无）'))

    # ---- 任务 1 ----
    print('=' * 92)
    print('任务 1 · 空间轮廓提取（对照 output/gt；取富化**前**的 _raw.jsonld，与分类器无关）')
    print('-' * 92)
    print('%-16s %8s %8s %9s %10s %9s %11s'
          % ('轮廓算法', '系统数', 'GT数', '数量比', '覆盖率', '一对一率', 'mIoU'))
    contour_out = {}

    def _contour_row(label: str, run: dict) -> None:
        keys = _keys_in(run['dir'], args.gt_dir, tuple(args.exclude))
        res = evaluate_contour(run, args.gt_dir, keys, args.min_iou)
        a = res['aggregate']
        contour_out[label] = res
        print('%-16s %8d %8d %9.3f %10.3f %9.3f %11.3f'
              % (label, a['n_sys'], a['n_gt'], a['count_ratio'],
                 a['coverage'], a['one_to_one'], a['miou_matched']))

    for r in contour_runs:
        _contour_row(r['method'], r)

    # 自检行：用 GT 轮廓去对 GT —— IoU/覆盖率应当 ≈1.0。
    # 它不参评，只用来证明"匹配 + 指标实现"本身是对的；若这行不是 ~1.0，说明评估代码有问题。
    if type_runs:
        _contour_row('GT(自检)', type_runs[0])
        print('注：`GT(自检)` 不参评 —— 它拿 GT 轮廓对 GT 自己，'
              '数值应≈1.0，用来验证评估实现无误。')

    # ---- 任务 2 ----
    print('\n' + '=' * 92)
    print('任务 2 · 空间类型识别（全部跑在 GT 轮廓上，保证同起点）')
    print('-' * 92)
    print('%-16s %8s %9s %10s %9s %12s'
          % ('分类器', 'GT空间', '准确率', '覆盖率', 'MacroF1', 'WeightedF1'))
    type_out = {}
    for r in type_runs:
        keys = _keys_in(r['dir'], args.gt_dir, tuple(args.exclude))
        res = evaluate_type(r, args.gt_dir, keys, label_space, args.min_iou)
        a = res['aggregate']
        type_out[r['method']] = res
        print('%-16s %8d %9.3f %10.3f %9.3f %12.3f'
              % (r['method'], a['n_gt'], a['accuracy'],
                 a['coverage'], a['macro_f1'], a['weighted_f1']))

    # ---- 端到端：任务 2 跑在任务 1 的轮廓上（3 × 3 排列组合）----
    e2e_out: dict = {}
    if e2e_runs:
        print('\n' + '=' * 92)
        print('端到端 · 任务 1 的轮廓 与 任务 2 的分类器 排列组合（任务 2 **跑在任务 1 的结果上**）')
        print('-' * 92)
        print('%-14s %-16s %8s %9s %9s %11s %11s'
              % ('轮廓算法', '分类器', 'GT空间', '轮廓覆盖', '条件准确', '端到端准确', 'MacroF1'))
        contour_order: list[str] = []
        for r in e2e_runs:
            keys = _keys_in(r['dir'], args.gt_dir, tuple(args.exclude))
            res = evaluate_type(r, args.gt_dir, keys, label_space, args.min_iou)
            bd = e2e_breakdown(res)
            a = res['aggregate']
            e2e_out[r['method']] = {
                'contour': r['contour'], 'clf': r['clf'],
                'aggregate': a, 'breakdown': bd,
                'per_drawing': res['per_drawing'],
                'confusion_matrix': res['confusion_matrix'],
                'n_drawings': res['n_drawings'],
            }
            if r['contour'] not in contour_order:
                contour_order.append(r['contour'])
            print('%-14s %-16s %8d %9.3f %9.3f %11.3f %11.3f'
                  % (r['contour'], r['clf'], bd['n_gt'], bd['contour_coverage'],
                     bd['accuracy_given_covered'], bd['accuracy'], a['macro_f1']))
        print('注：端到端准确率 = 轮廓覆盖 × 条件准确 ——  前者是任务 1 有没有把该 GT 空间找出来，')
        print('    后者是找到之后类型有没有判对。两者相乘才是真正端到端的数字。')

    payload = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'bench_dir': os.path.relpath(args.bench_dir, root),
        'excluded_prefixes': list(args.exclude),
        'gt_paths': gt_paths,
        'label_space': {'name': label_space.space, 'classes': list(label_space.classes),
                        # 把折叠规则一并写出：网页端要折叠**两边**的类别名才能直接对比
                        # （GT 里是 MainCorridor/SecondaryCorridor/PublicCorridor，
                        #   方法侧混着分类器输出与下游拓扑细化追加的类型）
                        'folding': dict(label_space.folding)},
        'contour_task': {
            m: {'aggregate': v['aggregate'], 'per_drawing': v['per_drawing'],
                'n_drawings': v['n_drawings']}
            for m, v in contour_out.items()},
        'type_task': {
            m: {'aggregate': v['aggregate'], 'per_drawing': v['per_drawing'],
                'confusion_matrix': v['confusion_matrix'], 'n_drawings': v['n_drawings']}
            for m, v in type_out.items()},
        'e2e_task': {
            m: {'contour': v['contour'], 'clf': v['clf'],
                'aggregate': v['aggregate'], 'breakdown': v['breakdown'],
                'per_drawing': v['per_drawing'],
                'confusion_matrix': v['confusion_matrix'],
                'n_drawings': v['n_drawings']}
            for m, v in e2e_out.items()},
    }

    if len(type_out) >= 2:
        print('\n--- 任务 2 逐 GT 类别 F1 ---')
        names = list(type_out)
        print('%-18s %s' % ('GT 类别', ''.join('%16s' % n[:15] for n in names)))
        for i, cls in enumerate(label_space.classes):
            first = type_out[names[0]]['aggregate']
            if not first['present'][i]:
                continue
            print('%-18s' % cls
                  + ''.join('%16.3f' % type_out[n]['aggregate']['f1_by_class'][i]
                            for n in names))

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or '.', exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=float)
        print('\n结果已写出: %s' % args.out_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
