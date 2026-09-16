"""从 CAD2Graph 的系统输出 + GT 图谱生成 SAGE-E 训练集（阶段 D）。

思路
----
SAGE-E 学的是"房间图 → 房间类型"，所以监督信号必须落在**系统自己提取出来的轮廓**上，
而不是 GT 的轮廓上 —— 否则训练/推理的输入分布不一致。

因此每张图纸做三件事：

1. 从 ``output/jsonld/<name>_raw.jsonld``（**富化前**的系统输出）建房间图；
   此时空间还没有 ``bldg:`` 类型，但已经有 ``bot:adjacentZone`` / ``bot:containsElement``；
2. 从 ``output/gt/<name>_gt.jsonld`` 取人工标注的空间类型；
3. 用**几何重叠**把系统轮廓对齐到 GT 轮廓，把 GT 的类型当作系统轮廓的标签。

第 3 步与 ``src/experiment/evaluator.py`` 的评估口径一致（都是"多对多几何重叠"），
所以训练标签与最终评估标签同源。

产物
----
``<out>.npz``
    ``node_feat / node_label / edge_index / edge_feat / graph_ptr / edge_ptr``
    （与 RoomGraph 导出同构，可直接喂给 ``scripts/train_sagee.py``）
``<out>.labels.json``
    类别表（**必须**与权重一起保存/加载）
``<out>.graphs.json``
    逐图元信息：来源文件、房间数、边数、被丢弃的节点数等（便于排查）

用法::

    python scripts/export_sagee_dataset.py \
        --jsonld-dir output/jsonld --gt-dir output/gt \
        --out data/cad2graph_sagee --space FULL
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spatial.sagee.graph import _as_polygon, from_jsonld        # noqa: E402
from src.spatial.sagee.labels import LabelSpace                  # noqa: E402

#: 系统输出 / GT 文件名里的 "族 + 序号"，用于配对
_PATTERN = re.compile(r'^(?P<family>[0-9]+suite|sample)(?:_annotated)?\s*\((?P<idx>\d+)\)')


def _key(path: str) -> str:
    """把 ``2suite (1)_raw.jsonld`` 与 ``2suite_annotated (1)_gt.jsonld`` 归一到同一个键。"""
    base = os.path.basename(path)
    m = _PATTERN.match(base)
    if m:
        return '%s#%s' % (m.group('family'), m.group('idx'))
    # 兜底：去掉已知后缀
    return re.sub(r'(_raw|_gt)?\.jsonld$', '', base)


def _load(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _gt_polygons(gt: dict) -> tuple[list[str], list[Polygon], list[str]]:
    """从 GT 图谱取出（id, 多边形, 类型名）。"""
    ids, polys, labels = [], [], []
    for node in gt.get('@graph') or []:
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
            from shapely.wkt import loads as wkt_loads
            poly = wkt_loads(wkt)
        except Exception:                                        # noqa: BLE001
            continue
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        label = next((t.split(':', 1)[1] for t in types if t.startswith('bldg:')), None)
        if label is None:
            continue
        ids.append(str(node.get('@id', '')))
        polys.append(poly)
        labels.append(label)
    return ids, polys, labels


def match_labels(sys_polys: list[Polygon], gt_ids: list[str], gt_polys: list[Polygon],
                 gt_labels: list[str], min_iou: float = 0.3) -> tuple[list[str | None], dict]:
    """把每个系统轮廓匹配到 IoU 最大的 GT 轮廓。

    先按 IoU 匹配；IoU 太小则退回"质心落在 GT 内"，再不行置 ``None``（该节点不参与训练）。
    """
    out: list[str | None] = [None] * len(sys_polys)
    how = {'iou': 0, 'centroid': 0, 'none': 0}
    for i, poly in enumerate(sys_polys):
        if poly is None or poly.is_empty:
            how['none'] += 1
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
            out[i] = gt_labels[best_j]
            how['iou'] += 1
            continue
        c = poly.centroid
        hit = next((j for j, gpoly in enumerate(gt_polys) if gpoly.contains(c)), None)
        if hit is not None:
            out[i] = gt_labels[hit]
            how['centroid'] += 1
        else:
            how['none'] += 1
    return out, how


def _sys_polygons(graph_dict: dict) -> list[Polygon | None]:
    """按 ``from_jsonld`` 的**同一顺序**取出系统轮廓的多边形。

    顺序必须与建图时一致，否则标签会挂到错误的节点上。两处都用
    "遍历 @graph、筛 bot:Space" 的朴素顺序，不依赖任何排序。
    """
    polys: list[Polygon | None] = []
    for node in graph_dict.get('@graph') or []:
        if not isinstance(node, dict):
            continue
        types = node.get('@type') or []
        types = [types] if isinstance(types, str) else list(types)
        if 'bot:Space' in types:
            polys.append(_as_polygon(node.get('geo:asWKT')))
    return polys


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='CAD2Graph -> SAGE-E 数据集')
    ap.add_argument('--jsonld-dir', default=os.path.join(root, 'output', 'jsonld'))
    ap.add_argument('--gt-dir', default=os.path.join(root, 'output', 'gt'))
    ap.add_argument('--out', default=os.path.join(root, 'data', 'cad2graph_sagee'))
    ap.add_argument('--space', default='FULL', choices=['FULL', 'CORE'])
    ap.add_argument('--min-samples', type=int, default=10,
                    help='CORE 空间下低于该样本数的整类会被丢弃')
    ap.add_argument('--min-iou', type=float, default=0.3)
    ap.add_argument('--min-nodes', type=int, default=3,
                    help='少于该房间数的图直接丢弃（多为解析失败的图纸）')
    args = ap.parse_args()

    # ---- 配对系统输出与 GT ------------------------------------------------- #
    sys_files = {_key(p): p for p in sorted(glob.glob(os.path.join(args.jsonld_dir, '*_raw.jsonld')))}
    gt_files = {_key(p): p for p in sorted(glob.glob(os.path.join(args.gt_dir, '*.jsonld')))}
    common = sorted(set(sys_files) & set(gt_files))
    print('系统输出 %d 份 | GT %d 份 | 可配对 %d 对'
          % (len(sys_files), len(gt_files), len(common)))
    if not common:
        print('[!!] 没有任何可配对的图纸，请检查 --jsonld-dir / --gt-dir')
        return 1

    # ---- 先扫一遍收集全部标签，才能定类别表 --------------------------------- #
    staged: list[tuple[str, object, list[str | None], dict]] = []
    all_gt_labels: list[str] = []
    match_totals = {'iou': 0, 'centroid': 0, 'none': 0}
    for key in common:
        graph_dict = _load(sys_files[key])
        gt = _load(gt_files[key])
        gt_ids, gt_polys, gt_labels = _gt_polygons(gt)
        all_gt_labels.extend(gt_labels)
        if not gt_polys:
            print('  [--] %-22s GT 无可用空间，跳过' % key)
            continue
        room_graph = from_jsonld(graph_dict, source=os.path.basename(sys_files[key]))
        if room_graph.n_nodes < args.min_nodes:
            print('  [--] %-22s 系统仅 %d 个空间，跳过' % (key, room_graph.n_nodes))
            continue
        sys_polys = _sys_polygons(graph_dict)
        labels, how = match_labels(sys_polys, gt_ids, gt_polys, gt_labels, args.min_iou)
        for k in match_totals:
            match_totals[k] += how[k]
        if len(labels) != room_graph.n_nodes:
            print('  [!!] %-22s 轮廓数(%d)与节点数(%d)不一致，跳过'
                  % (key, len(labels), room_graph.n_nodes))
            continue
        staged.append((key, room_graph, labels, how))

    total_sys = sum(match_totals.values())
    print('  匹配情况: IoU %d (%.1f%%) | 质心回退 %d (%.1f%%) | 未匹配 %d (%.1f%%)'
          % (match_totals['iou'], 100.0 * match_totals['iou'] / max(1, total_sys),
             match_totals['centroid'], 100.0 * match_totals['centroid'] / max(1, total_sys),
             match_totals['none'], 100.0 * match_totals['none'] / max(1, total_sys)))

    if not staged:
        print('[!!] 没有可用图纸')
        return 1

    # 标签空间在**全部 GT 标签**上拟合，而不是只在匹配成功的子集上 ——
    # 否则匹配一差，类别表就会静默缩水，训练/推理的类别下标跟着错位。
    label_space = LabelSpace.fit(all_gt_labels, space=args.space)
    print('\n标签空间: %s -> %d 类  %s'
          % (label_space.space, label_space.n_class, list(label_space.classes)))

    if label_space.space == 'CORE':
        # 第二遍：类别表是用 GT 标签定的，但真正决定能不能学的是**导出数据里**
        # 各类的样本数 —— 轮廓提取造不出来的类别，在训练集里就几个样本。
        tally: dict[str, int] = {}
        for _stage_key, _rg, labels, _how in staged:
            for lb in labels:
                if not lb:
                    continue
                idx = label_space.encode(lb)
                if idx is not None:
                    name = label_space.decode(idx)
                    tally[name] = tally.get(name, 0) + 1
        starving = sorted(n for n, c in tally.items() if c < args.min_samples)
        if starving:
            label_space = label_space.with_classes(
                n for n in tally if n not in set(starving))
            print('CORE 丢弃样本 < %d 的类别: %s'
                  % (args.min_samples, ', '.join('%s(%d)' % (n, tally[n]) for n in starving)))
        print('最终标签空间: %d 类  %s'
              % (label_space.n_class, list(label_space.classes)))

    # ---- 编码并写盘 --------------------------------------------------------- #
    node_feat, node_label = [], []
    edge_index, edge_feat = [], []
    graph_ptr, edge_ptr = [0], [0]
    dropped = 0
    details = []

    for key, room_graph, labels, how in staged:
        keep = [i for i, lb in enumerate(labels) if lb and label_space.encode(lb) is not None]
        if len(keep) < args.min_nodes:
            print('  [--] %-22s 标签可用节点仅 %d 个，跳过' % (key, len(keep)))
            continue
        remap = {old: new for new, old in enumerate(keep)}
        dropped += room_graph.n_nodes - len(keep)

        base = graph_ptr[-1]
        node_feat.append(room_graph.node_feat[keep])
        node_label.append(np.asarray([label_space.encode(labels[i]) for i in keep], dtype=np.int64))

        ei = room_graph.edge_index
        sel = [k for k in range(ei.shape[1]) if int(ei[0, k]) in remap and int(ei[1, k]) in remap]
        if sel:
            sub = ei[:, sel]
            # ⚠️ npz 里存的是**全局**节点索引（读取时会减去 graph_ptr 偏移变成局部），
            # 所以这里必须把 remap 出来的局部下标再加上本图的 base。
            edge_index.append(np.asarray([
                [base + remap[int(sub[0, p])] for p in range(len(sel))],
                [base + remap[int(sub[1, p])] for p in range(len(sel))],
            ], dtype=np.int64))
            edge_feat.append(room_graph.edge_feat[sel])
        else:
            edge_index.append(np.zeros((2, 0), dtype=np.int64))
            edge_feat.append(np.zeros((0, room_graph.edge_feat.shape[1] or 5), dtype=np.float64))

        graph_ptr.append(base + len(keep))
        edge_ptr.append(edge_ptr[-1] + edge_index[-1].shape[1])

        details.append({
            'key': key,
            'sys_file': room_graph.source,
            'n_nodes': len(keep),
            'n_nodes_dropped': room_graph.n_nodes - len(keep),
            'n_edges': int(edge_index[-1].shape[1]),
            'match': how,
            'label_counts': dict(zip(*[x.tolist() for x in np.unique(
                [label_space.decode(int(v)) for v in node_label[-1]], return_counts=True)])),
        })

    arrays = {
        'node_feat': np.concatenate(node_feat, axis=0),
        'node_label': np.concatenate(node_label, axis=0),
        'edge_index': (np.concatenate(edge_index, axis=1) if any(e.shape[1] for e in edge_index)
                       else np.zeros((2, 0), dtype=np.int64)),
        'edge_feat': np.concatenate(edge_feat, axis=0),
        'graph_ptr': np.asarray(graph_ptr, dtype=np.int64),
        'edge_ptr': np.asarray(edge_ptr, dtype=np.int64),
    }

    out_npz = args.out + '.npz'
    os.makedirs(os.path.dirname(out_npz) or '.', exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    label_space.save(args.out + '.labels.json')
    with open(args.out + '.graphs.json', 'w', encoding='utf-8') as f:
        json.dump({'space': label_space.space, 'graphs': details}, f, indent=2, ensure_ascii=False)

    print('\n导出完成: %s' % out_npz)
    print('  图数        : %d' % (len(graph_ptr) - 1))
    print('  节点数      : %d' % arrays['node_feat'].shape[0])
    print('  边数        : %d' % arrays['edge_index'].shape[1])
    print('  特征维度    : node=%d edge=%d'
          % (arrays['node_feat'].shape[1], arrays['edge_feat'].shape[1]))
    print('  类别数      : %d  %s' % (label_space.n_class, list(label_space.classes)))
    print('  丢弃的节点  : %d（无 GT 标签或不在本标签空间）' % dropped)
    counts = np.bincount(arrays['node_label'], minlength=label_space.n_class)
    print('  类别分布    :')
    for i, c in enumerate(counts):
        print('    %-20s %4d  (%.1f%%)'
              % (label_space.classes[i], c, 100.0 * c / max(1, counts.sum())))
    print('  labels      : %s' % (args.out + '.labels.json'))
    print('  graphs      : %s' % (args.out + '.graphs.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
