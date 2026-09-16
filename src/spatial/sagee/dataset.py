"""SAGE-E 的数据集读取与指标计算 —— **不依赖 torch / DGL / shapely**。

``train_sagee.py``（torch 训练）与 ``scripts/check_sagee_port.py``（numpy 回归测试）
共用这里的实现，保证两边用的是同一套划分与同一套指标定义。

⚠️ 本模块刻意不 import ``src.spatial`` 的 ``__init__``（那会连带拉起 shapely 等
轮廓算法依赖），因此可以跑在任何只有 numpy 的环境里。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ['NpzGraphDataset', 'split_indices', 'confusion', 'macro_f1', 'weighted_f1',
           'per_class_report']


class NpzGraphDataset:
    """把扁平化的 ``.npz``（见 ``scripts/export_roomgraph.py``）切成一堆小图。

    ``get(i)`` 返回纯 numpy 数组 ``(node_feat, edge_index, edge_feat, node_label)``，
    其中 ``edge_index`` 已就地重新编号（从 0 起），可直接喂给任一实现。
    """

    def __init__(self, npz_path: str, indices: Sequence[int] | None = None):
        with np.load(npz_path) as data:
            self.node_feat = data['node_feat']
            self.node_label = data['node_label']
            self.edge_index = data['edge_index']
            self.edge_feat = data['edge_feat']
            self.graph_ptr = data['graph_ptr']
            self.edge_ptr = data['edge_ptr']
        self.n_graph_total = int(self.graph_ptr.shape[0] - 1)
        self.indices = (list(range(self.n_graph_total)) if indices is None
                        else [int(i) for i in indices])

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get(i)

    def get(self, i: int):
        g = self.indices[i]
        ns, ne = int(self.graph_ptr[g]), int(self.graph_ptr[g + 1])
        es, ee = int(self.edge_ptr[g]), int(self.edge_ptr[g + 1])
        return (
            np.ascontiguousarray(self.node_feat[ns:ne]),
            np.ascontiguousarray(self.edge_index[:, es:ee] - ns, dtype=np.int64),
            np.ascontiguousarray(self.edge_feat[es:ee]),
            np.ascontiguousarray(self.node_label[ns:ne], dtype=np.int64),
        )

    @property
    def dim_node(self) -> int:
        return int(self.node_feat.shape[1])

    @property
    def dim_edge(self) -> int:
        return int(self.edge_feat.shape[1])

    @property
    def n_class(self) -> int:
        return int(self.node_label.max()) + 1

    def n_nodes(self) -> int:
        return int(sum(self.graph_ptr[i + 1] - self.graph_ptr[i] for i in self.indices))

    def gather(self, key: str) -> np.ndarray:
        """把本子集内所有图的某个特征矩阵纵向拼接起来。

        节点类用 ``node_feat``（按 ``graph_ptr`` 切）；
        边类用 ``edge_feat``（按 ``edge_ptr`` 切）。
        """
        arr = getattr(self, key)
        ptr = self.edge_ptr if key.startswith('edge') else self.graph_ptr
        parts = [arr[int(ptr[g]):int(ptr[g + 1])] for g in self.indices]
        if not parts:
            return np.zeros((0, arr.shape[1]), dtype=arr.dtype)
        return np.concatenate(parts, axis=0)


def split_indices(n_graph: int, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """逐位复现作者 notebook 的划分。

    作者用的是 ``sklearn.model_selection.train_test_split``：

        trainvalid, test = train_test_split(bg, test_size=0.2, random_state=42)
        train, valid     = train_test_split(trainvalid, test_size=0.1, random_state=42)

    它的内部实现就是 ``RandomState(seed).permutation(n)`` 后按 ``ceil(n*ratio)`` 切分，
    所以用 numpy 即可逐位复现，无需引入 sklearn 依赖。
    """
    perm = np.random.RandomState(seed).permutation(n_graph)
    n_test = int(np.ceil(n_graph * 0.2))
    test, trainvalid = perm[:n_test], perm[n_test:]
    perm2 = np.random.RandomState(seed).permutation(len(trainvalid))
    n_valid = int(np.ceil(len(trainvalid) * 0.1))
    valid, train = trainvalid[perm2[:n_valid]], trainvalid[perm2[n_valid:]]
    return sorted(train.tolist()), sorted(valid.tolist()), sorted(test.tolist())


def confusion(pred: np.ndarray, gt: np.ndarray, n_class: int) -> np.ndarray:
    """行=真实，列=预测。"""
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    np.add.at(cm, (gt, pred), 1)
    return cm


def _f1_per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0), cm.sum(1).astype(np.float64)


def macro_f1(cm: np.ndarray) -> float:
    """只在**出现过的类别**上取平均（长尾数据集里没有样本的类别不该拖低指标）。"""
    f1, support = _f1_per_class(cm)
    present = support > 0
    return float(f1[present].mean()) if present.any() else 0.0


def weighted_f1(cm: np.ndarray) -> float:
    f1, support = _f1_per_class(cm)
    return float((f1 * support).sum() / max(support.sum(), 1e-12))


def per_class_report(cm: np.ndarray, class_names: Sequence[str] | None = None) -> list[dict]:
    """逐类 precision / recall / F1 / support，供对比表使用。"""
    f1, support = _f1_per_class(cm)
    tp = np.diag(cm).astype(np.float64)
    prec = np.where(cm.sum(0) > 0, tp / np.maximum(cm.sum(0), 1e-12), 0.0)
    rec = np.where(support > 0, tp / np.maximum(support, 1e-12), 0.0)
    out = []
    for i in range(cm.shape[0]):
        out.append({
            'index': i,
            'name': (class_names[i] if class_names and i < len(class_names) else 'class_%d' % i),
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        })
    return out
