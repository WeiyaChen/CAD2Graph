"""审计：系统产物里出现的 ``bldg:`` 类型，有多少落在标签空间内。

要回答的问题
------------
评估时我们把系统空间的类型过一遍折叠规则（``fold_label``），再查标签空间。
**落在标签空间之外的类型会被记成错误**（``<空间外>``）。所以必须知道：

1. 系统实际会产出哪些类型？
2. 其中哪些不在标签空间里（⇒ 被冤枉地判错）？
3. ``@type`` 里同时挂多个类型时，评估只取第 0 个 —— 这个选择合不合理？

用法::

    python scripts/audit_label_vocabulary.py output/holdout_thr000 \\
        --label-space data/sagee_cad2graph_core.labels.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spatial.sagee.labels import LabelSpace, fold_label   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description='审计系统产物的类型词汇')
    ap.add_argument('dirs', nargs='+', help='一个或多个系统产物目录')
    ap.add_argument('--label-space', required=True)
    args = ap.parse_args()

    space = LabelSpace.load(args.label_space)
    rules = space.folding
    inside = set(space.classes)

    for d in args.dirs:
        raw = collections.Counter()
        folded = collections.Counter()
        first_slot = collections.Counter()
        n_multi = 0
        n_space = 0
        for path in sorted(glob.glob(os.path.join(d, '*.jsonld'))):
            if path.endswith('_raw.jsonld'):
                continue
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            for node in data.get('@graph') or []:
                if not isinstance(node, dict):
                    continue
                types = node.get('@type') or []
                types = [types] if isinstance(types, str) else list(types)
                if 'bot:Space' not in types:
                    continue
                names = [t.split(':', 1)[1] for t in types if t.startswith('bldg:')]
                if not names:
                    continue
                n_space += 1
                if len(names) > 1:
                    n_multi += 1
                for nm in names:
                    raw[nm] += 1
                    folded[fold_label(nm, rules)] += 1
                first_slot[fold_label(names[0], rules)] += 1

        print('=' * 74)
        print('目录: %s' % d)
        print('  有类型的空间 %d 个，其中 @type 挂多个 bldg: 类型的 %d 个 (%.1f%%)'
              % (n_space, n_multi, 100.0 * n_multi / max(1, n_space)))
        print('\n  原始类型 -> 折叠后（* = 不在标签空间内，会被判为「空间外」）')
        print('  %-22s %7s  ->  %-20s %s' % ('原始', '次数', '折叠后', ''))
        for nm, c in raw.most_common():
            f = fold_label(nm, rules)
            why = '' if f in inside else '  *** 不在标签空间 -> 算错 ***'
            print('  %-22s %7d  ->  %-20s%s' % (nm, c, f, why))

        outside = {f: c for f, c in folded.items() if f not in inside}
        n_out = sum(outside.values())
        print('\n  所有类型位（含同一个空间的多个类型）: %d，其中落标签空间外 %d (%.1f%%)'
              % (sum(folded.values()), n_out, 100.0 * n_out / max(1, sum(folded.values()))))
        out_first = {f: c for f, c in first_slot.items() if f not in inside}
        n_out_first = sum(out_first.values())
        print('  只取 @type[0] 时（= 当前评估口径）: %d，其中落标签空间外 %d (%.1f%%)'
              % (n_space, n_out_first, 100.0 * n_out_first / max(1, n_space)))
        for f, c in sorted(out_first.items(), key=lambda kv: -kv[1]):
            print('     %-22s %5d' % (f, c))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
