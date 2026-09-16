"""把 SAGE-E 的 RoomGraph 数据集导出为中立格式 ``.npz``（一次性工具）。

`roomgraph.bin` 是 DGL 的私有二进制格式，只有 ``dgl.load_graphs`` 能读。为了让
CAD2Graph 侧**不依赖 DGL / torch**，这里把它转成一个纯 numpy 能读的 ``.npz``：

    node_feat   (N, 8)   float64   节点特征
    node_label  (N, )    int64     标签（已 argmax，不是 one-hot）
    edge_index  (2, E)   int64     全局节点索引
    edge_feat   (E, 5)   int64     边特征
    graph_ptr   (G+1, )  int64     第 g 张图的节点区间 [ptr[g], ptr[g+1])
    edge_ptr    (G+1, )  int64     第 g 张图的边区间 [ptr[g], ptr[g+1])

顺带写一个 ``<out>.meta.json``，记录来源与许可，便于溯源。

⚠️ DGL 只在本脚本中需要，而且只在**导出时**需要一次。若当前环境没装 dgl，
用 ``--dgl-path`` 指向一个临时安装目录即可。

用法::

    python scripts/export_roomgraph.py \
        --bin third_party/SAGE-E/dataset/roomgraph.bin \
        --out data/roomgraph.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _load_graphs(path: str, dgl_path: str | None):
    if dgl_path:
        sys.path.insert(0, dgl_path)
    try:
        from dgl.data.utils import load_graphs
    except ImportError as exc:                                   # pragma: no cover
        raise SystemExit(
            'DGL 不可用（%s）。\n'
            '本脚本只在导出时需要 DGL。任选其一：\n'
            '  1) pip install dgl torchdata pydantic\n'
            '  2) python scripts/export_roomgraph.py --dgl-path <装了 dgl 的目录>'
            % exc
        ) from exc
    return load_graphs(path)[0]


def export(bin_path: str, out_path: str, dgl_path: str | None = None) -> dict:
    graphs = _load_graphs(bin_path, dgl_path)
    n_graph = len(graphs)

    node_feat, node_label = [], []
    edge_index, edge_feat = [], []
    graph_ptr, edge_ptr = [0], [0]

    for g in graphs:
        feat = np.asarray(g.ndata['feat'].numpy(), dtype=np.float64)
        lab = np.asarray(g.ndata['label'].numpy())
        label = np.argmax(lab, axis=1).astype(np.int64) if lab.ndim == 2 else lab.astype(np.int64)
        src, dst = g.edges()
        src = np.asarray(src.numpy(), dtype=np.int64)
        dst = np.asarray(dst.numpy(), dtype=np.int64)
        rel = np.asarray(g.edata['relation'].numpy(), dtype=np.int64)

        base = graph_ptr[-1]
        node_feat.append(feat)
        node_label.append(label)
        edge_index.append(np.stack([src + base, dst + base]))
        edge_feat.append(rel)
        graph_ptr.append(base + feat.shape[0])
        edge_ptr.append(edge_ptr[-1] + rel.shape[0])

    arrays = {
        'node_feat': np.concatenate(node_feat, axis=0),
        'node_label': np.concatenate(node_label, axis=0),
        'edge_index': np.concatenate(edge_index, axis=1),
        'edge_feat': np.concatenate(edge_feat, axis=0),
        'graph_ptr': np.asarray(graph_ptr, dtype=np.int64),
        'edge_ptr': np.asarray(edge_ptr, dtype=np.int64),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    n_class = int(arrays['node_label'].max()) + 1
    meta = {
        'source': os.path.basename(bin_path),
        'generator': 'DGL.save_graphs / dgl.data.utils.load_graphs',
        'n_graphs': int(n_graph),
        'n_nodes': int(arrays['node_feat'].shape[0]),
        'n_edges': int(arrays['edge_feat'].shape[0]),
        'node_feat_dim': int(arrays['node_feat'].shape[1]),
        'edge_feat_dim': int(arrays['edge_feat'].shape[1]),
        'n_classes': n_class,
        'class_counts': np.bincount(arrays['node_label'], minlength=n_class).tolist(),
        'license': 'MIT (ZijianWang-ZW/SAGE-E)',
        'citation': (
            'Wang, Z., Sacks, R., Yeung, T., 2022. Exploring graph neural networks '
            'for semantic enrichment: Room type classification. '
            'Automation in Construction 134:104039.'
        ),
    }
    with open(os.path.splitext(out_path)[0] + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_bin = os.path.join(here, '..', 'third_party', 'SAGE-E', 'dataset', 'roomgraph.bin')

    ap = argparse.ArgumentParser(description='RoomGraph (.bin) -> .npz')
    ap.add_argument('--bin', default=default_bin, help='roomgraph.bin 路径')
    ap.add_argument('--out', default='data/roomgraph.npz', help='输出 .npz 路径')
    ap.add_argument('--dgl-path', default=None, help='额外的 sys.path（临时安装的 dgl 目录）')
    args = ap.parse_args()

    meta = export(args.bin, args.out, args.dgl_path)

    print('导出完成: %s' % args.out)
    for k in ('n_graphs', 'n_nodes', 'n_edges', 'node_feat_dim',
              'edge_feat_dim', 'n_classes'):
        print('  %-14s %s' % (k, meta[k]))
    print('  %-14s %s' % ('class_counts', meta['class_counts']))
    print('  meta        %s' % (os.path.splitext(args.out)[0] + '.meta.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
