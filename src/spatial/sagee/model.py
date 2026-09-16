"""SAGE-E 的纯 numpy 前向（推理侧，**不需要 torch / DGL**）。

为什么可以这么做
----------------
SAGE-E 的每一层只有两个线性层加一次"按边求和"：

    agg[v] = Σ_{u→v} ReLU(W_msg [h_u ‖ e_uv] + b_msg)
    h_v    = ReLU(W_apply [h_v ‖ agg[v]] + b_apply)

DGL 在这套模型里**只承担消息传递**，而小图（户型一般 5–13 个房间）用
``np.add.at`` 就足够快了。把它换成 numpy 之后：

* CAD2Graph 的 ``cadruler`` 环境不需要装 torch，也不需要 DGL；
* 推理可以直接在进程内完成，不必像 VecFloorSeg 那样开子进程；
* 权重只需一次性从 ``.pt`` 转成 ``.npz``（见 ``scripts/train_sagee.py --export-weights``）。

⚠️ 保持与作者实现一致的两个细节：
1. **不加自环**（作者 notebook 里 ``add_self_loop`` 是注释掉的），
   所以一个没有入边的节点聚合结果为全 0，而不是它自己；
2. ``fn.sum`` 是**对入边求和**，即按 ``edge_index[1]``（dst）聚合。
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

__all__ = ['SageeNumpy', 'load_weights', 'load_norm', 'run_sagee']

#: 作者采用 4 层、隐藏维 (50, 50, 25)
HIDDEN_DIMS = (50, 50, 25)


class SageeNumpy:
    """从 ``.npz`` 权重加载的 SAGE-E 推理模型（纯 numpy）。

    Args:
        weights: ``.npz`` 路径或已载入的字典。
        norm: 逐列 z-score 参数（``.norm.json`` 路径或字典）。**必须与训练时一致** ——
            用了 ``--normalize`` 训练的模型不传 norm 会得到完全错误的输入分布。
            传 ``None`` 且存在同名的 ``<weights>.norm.json`` 时会自动载入。
        n_class: 覆盖输出类别数（默认从权重形状推导）。
    """

    def __init__(self, weights: str | dict[str, np.ndarray],
                 norm: str | dict[str, Any] | None = None,
                 n_class: int | None = None):
        sd = load_weights(weights) if isinstance(weights, str) else dict(weights)
        self.state = {k: np.asarray(v, dtype=np.float64) for k, v in sd.items()}
        self.n_layer = 1 + max(
            int(k.split('__')[1]) for k in self.state if k.startswith('layers__')
        )
        self.n_class = int(n_class or self.state['layers__%d__W_msg__bias' % (self.n_layer - 1)].shape[0])
        self.dropout_prob = 0.0      # 推理恒为 0

        # 提前校验维度自洽，避免运行期才炸
        self.ndim_in = int(self.state['layers__0__W_apply__weight'].shape[1]
                           - self.state['layers__0__W_msg__weight'].shape[0])
        self.edim = int(self.state['layers__0__W_msg__weight'].shape[1] - self.ndim_in)

        if norm is None and isinstance(weights, str):
            sidecar = os.path.splitext(weights)[0] + '.norm.json'
            norm = sidecar if os.path.isfile(sidecar) else None
        self.norm = load_norm(norm) if norm is not None else None
        if self.norm is not None:
            if len(self.norm['node_mean']) != self.ndim_in or len(self.norm['edge_mean']) != self.edim:
                raise ValueError('norm 的维度 (%d/%d) 与权重 (%d/%d) 不一致'
                                 % (len(self.norm['node_mean']), len(self.norm['edge_mean']),
                                    self.ndim_in, self.edim))

    # ------------------------------------------------------------------ forward
    def forward(self, node_feat: np.ndarray, edge_index: np.ndarray,
                edge_feat: np.ndarray) -> np.ndarray:
        """返回每个节点的 logits，形状 ``(N, n_class)``。

        Args:
            node_feat: ``(N, ndim_in)``
            edge_index: ``(2, E)``，第 0 行是 src，第 1 行是 dst（已就地编号，从 0 起）
            edge_feat: ``(E, edim)``
        """
        h = np.asarray(node_feat, dtype=np.float64)
        ei = np.asarray(edge_index, dtype=np.int64)
        e = np.asarray(edge_feat, dtype=np.float64)
        if self.norm is not None:
            h = (h - self.norm['node_mean']) / self.norm['node_std']
            if e.shape[0]:
                e = (e - self.norm['edge_mean']) / self.norm['edge_std']
        if h.ndim != 2 or h.shape[1] != self.ndim_in:
            raise ValueError('node_feat 期望 (N, %d)，实得 %s' % (self.ndim_in, h.shape))
        if e.ndim != 2 or e.shape[1] != self.edim:
            raise ValueError('edge_feat 期望 (E, %d)，实得 %s' % (self.edim, e.shape))
        if ei.size and (ei.shape[0] != 2 or e.shape[0] != ei.shape[1]):
            raise ValueError('edge_index 应为 (2,E) 且与 edge_feat 行数一致，'
                             '实得 %s / %s' % (ei.shape, e.shape))

        src, dst = ei[0], ei[1]
        for i in range(self.n_layer):
            w_msg = self.state['layers__%d__W_msg__weight' % i]
            b_msg = self.state['layers__%d__W_msg__bias' % i]
            w_app = self.state['layers__%d__W_apply__weight' % i]
            b_app = self.state['layers__%d__W_apply__bias' % i]

            out_dim = w_msg.shape[0]
            if e.shape[0]:
                cat = np.concatenate([h[src], e], axis=1)
                msg = np.maximum(cat @ w_msg.T + b_msg, 0.0)
                agg = np.zeros((h.shape[0], out_dim), dtype=np.float64)
                np.add.at(agg, dst, msg)          # == fn.sum('m', 'h_neigh')
            else:
                agg = np.zeros((h.shape[0], out_dim), dtype=np.float64)

            h = np.maximum(np.concatenate([h, agg], axis=1) @ w_app.T + b_app, 0.0)
        return h

    def predict(self, node_feat, edge_index, edge_feat) -> np.ndarray:
        """返回每个节点的预测类别（int 数组）。"""
        return self.forward(node_feat, edge_index, edge_feat).argmax(axis=1)

    def predict_proba(self, node_feat, edge_index, edge_feat) -> np.ndarray:
        """返回每个节点的 softmax 概率，形状 ``(N, n_class)``。"""
        logits = self.forward(node_feat, edge_index, edge_feat)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def load_weights(weights: str | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """读取 ``.npz`` 权重；键名里的 ``__`` 是 ``.`` 的替身（npz 键不能含点）。"""
    if isinstance(weights, dict):
        return {k: np.asarray(v) for k, v in weights.items()}
    if not os.path.isfile(weights):
        raise FileNotFoundError('SAGE-E 权重不存在: %s\n'
                                '请先用 scripts/train_sagee.py --export-weights 生成。' % weights)
    with np.load(weights) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def load_norm(norm: str | dict[str, Any]) -> dict[str, np.ndarray]:
    """读取 z-score 参数（``.norm.json`` 或已解析的字典），统一转成 numpy 数组。"""
    if isinstance(norm, str):
        if not os.path.isfile(norm):
            raise FileNotFoundError('标准化参数不存在: %s' % norm)
        with open(norm, encoding='utf-8') as f:
            norm = json.load(f)
    return {
        'node_mean': np.asarray(norm['node_mean'], dtype=np.float64),
        'node_std': np.asarray(norm['node_std'], dtype=np.float64),
        'edge_mean': np.asarray(norm['edge_mean'], dtype=np.float64),
        'edge_std': np.asarray(norm['edge_std'], dtype=np.float64),
    }


def weights_meta_path(weights_path: str) -> str:
    return os.path.splitext(weights_path)[0] + '.meta.json'


def load_weights_meta(weights_path: str) -> dict[str, Any]:
    path = weights_meta_path(weights_path)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def run_sagee(weights: str | dict[str, np.ndarray], node_feat, edge_index,
              edge_feat) -> np.ndarray:
    """一次性的便捷入口：加载权重 → 前向 → 返回 logits。"""
    return SageeNumpy(weights).forward(node_feat, edge_index, edge_feat)
