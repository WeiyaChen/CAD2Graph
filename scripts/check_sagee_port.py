"""阶段 A 回归测试：验证纯 numpy 的 SAGE-E 端口与 torch 版逐位一致。

不需要 torch / DGL —— 只读 ``.npz`` 权重与 ``.npz`` 数据集。
基准是 torch 版用 ``best_user.pt`` 跑出的结果：

    Accuracy 0.7928 | Macro F1 0.7801 | 混淆矩阵 81 个格全部一致

只要这个脚本还能打印出同样的混淆矩阵，就说明 numpy 端口没有退化。

用法::

    python scripts/check_sagee_port.py \
        --dataset data/roomgraph.npz --weights data/sagee_roomgraph_C9.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spatial.sagee.dataset import (                      # noqa: E402
    NpzGraphDataset,
    confusion,
    macro_f1,
    split_indices,
    weighted_f1,
)
from src.spatial.sagee.model import SageeNumpy               # noqa: E402

#: torch 版用 best_user.pt 的基准结果（见 docs/sagee_baseline_plan.md §3.4）
EXPECTED_CM = np.array([
    [35,  0,  9,  0,  0,  0,  0,  0,  1],
    [ 1, 44,  0,  0,  0,  0,  0,  0,  0],
    [10,  0, 34,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, 72,  6,  3,  0,  0,  0],
    [ 0,  0,  0,  3, 27,  2,  0,  0,  0],
    [ 1,  0,  0, 17,  0, 46,  3,  0,  3],
    [ 0,  0,  0,  2,  0, 17,  8,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0, 41,  0],
    [ 0,  0,  0,  4,  0,  4,  0,  0, 22],
])
EXPECTED_ACC = 0.7928
EXPECTED_F1 = 0.7801


def main() -> int:
    ap = argparse.ArgumentParser(description='SAGE-E numpy 端口回归测试')
    ap.add_argument('--dataset', default='data/roomgraph.npz')
    ap.add_argument('--weights', default='data/sagee_roomgraph_C9.npz')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    ds_all = NpzGraphDataset(args.dataset)
    _, _, test_idx = split_indices(ds_all.n_graph_total, seed=args.seed)
    test_ds = NpzGraphDataset(args.dataset, test_idx)
    n_class = ds_all.n_class

    model = SageeNumpy(args.weights)
    print('权重        : %s' % args.weights)
    print('结构        : %d 层, ndim_in=%d, edim=%d, n_class=%d'
          % (model.n_layer, model.ndim_in, model.edim, model.n_class))

    preds, gts = [], []
    for nf, ei, ef, lb in test_ds:
        logits = model.forward(nf, ei, ef)
        if not np.isfinite(logits).all():
            print('  [!!] 出现非有限值（图 %d）' % len(preds))
            return 1
        preds.append(logits.argmax(axis=1))
        gts.append(lb)
    pred, gt = np.concatenate(preds), np.concatenate(gts)

    cm = confusion(pred, gt, n_class)
    acc, mf1, wf1 = float((pred == gt).mean()), macro_f1(cm), weighted_f1(cm)

    print('\n测试图数    : %d   节点数: %d' % (len(test_ds), len(pred)))
    print('Accuracy    : %.4f   (torch 基准 %.4f)' % (acc, EXPECTED_ACC))
    print('Macro F1    : %.4f   (torch 基准 %.4f)' % (mf1, EXPECTED_F1))
    print('Weighted F1 : %.4f' % wf1)
    print('Confusion matrix (行=真实, 列=预测):')
    print(cm)

    ok = True
    if cm.shape != EXPECTED_CM.shape or not np.array_equal(cm, EXPECTED_CM):
        diff = (cm - EXPECTED_CM) if cm.shape == EXPECTED_CM.shape else None
        ok = False
        print('\n[!!] 混淆矩阵与基准不一致')
        if diff is not None:
            print('     差异格数: %d' % int((diff != 0).sum()))
    else:
        print('\n[ok] 混淆矩阵与 torch 版逐位一致（%d 个格全部相同）' % cm.size)
    if abs(mf1 - EXPECTED_F1) > 5e-4:
        ok = False
        print('[!!] Macro F1 偏差 %.4f 超过容差' % abs(mf1 - EXPECTED_F1))
    else:
        print('[ok] Macro F1 与基准一致（%.4f）' % mf1)

    print('\n' + '=' * 62)
    print('阶段 A 通过：numpy 端口与 torch 实现语义等价。' if ok else '阶段 A 未通过。')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
