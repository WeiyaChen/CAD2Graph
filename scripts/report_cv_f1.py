"""从 5-fold CV 的混淆矩阵看逐类 F1（argmax 强制选择，无弃权）。

用途：当线上评估出现某个类别 F1=0（大量弃权）时，用它区分两种原因：
* 这里也接近 0  -> **模型本身分不出来**（特征/标签问题）
* 这里还不错    -> 只是**置信度阈值**把它砍掉了（调 min_confidence 即可）

用法::

    python scripts/report_cv_f1.py --cv-json data/sagee_cad2graph_core.cv.json \
        --labels data/sagee_cad2graph_core.labels.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='CV 混淆矩阵的逐类 F1（无弃权）')
    ap.add_argument('--cv-json', default=os.path.join(root, 'data', 'sagee_cad2graph_core.cv.json'))
    ap.add_argument('--labels', default=os.path.join(root, 'data', 'sagee_cad2graph_core.labels.json'))
    args = ap.parse_args()

    with open(args.cv_json, encoding='utf-8') as f:
        cv = json.load(f)
    with open(args.labels, encoding='utf-8') as f:
        classes = json.load(f)['classes']

    cm = np.asarray(cv['confusion_matrix'], dtype=np.int64)
    print('CV: %d-fold  合并后 Accuracy %.4f  Macro F1 %.4f'
          % (cv.get('cv', 1), cv['pooled_accuracy'], cv['pooled_macro_f1']))
    print('混淆矩阵 %s，类别表 %d 个\n' % (cm.shape, len(classes)))

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    prec = np.where(cm.sum(0) > 0, tp / np.maximum(cm.sum(0), 1e-12), 0.0)
    rec = np.where(cm.sum(1) > 0, tp / np.maximum(cm.sum(1), 1e-12), 0.0)

    order = np.argsort(-cm.sum(1))
    print('%-18s %7s %8s %8s %8s   %s' % ('类别', '样本', 'Precision', 'Recall', 'F1', '主要误判'))
    for i in order:
        n = int(cm[i].sum())
        if n == 0:
            continue
        others = sorted(((cm[i, j], classes[j]) for j in range(cm.shape[1]) if j != i),
                        reverse=True)[:2]
        top = ', '.join('%s(%d)' % (c, k) for k, c in others if k)
        print('%-18s %7d %8.3f %8.3f %8.3f   %s'
              % (classes[i], n, prec[i], rec[i], f1[i], top))

    present = cm.sum(1) > 0
    print('\nMacro F1 (无弃权，argmax 强制选择): %.4f' % f1[present].mean())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
