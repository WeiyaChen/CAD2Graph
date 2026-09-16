"""对比多个分类器产物目录里的**空间数 / 构件数**，用来回答：

> 「同一张图纸、同样的轮廓算法，不同分类器产出的空间集合一样吗？」

结论是**不一样**（详见 `docs/sagee_baseline_plan.md` §6.5）：

* `TextMatching` / `LLMMultiStage` 可能给一个空间打**多个类型** → 被判为复合空间
  → 触发 ACD 切分 → 空间变多变小；
* `SAGEE` 是单标签节点分类器 → 永远不复合 → 永不切分。

所以按「系统空间」统计的对比表里，各分类器的**分母不同**，不能直接比百分比。
要么用 `scripts/evaluate_per_gt_space.py`（按 GT 空间统计），要么至少在报告里标注。

2. 「换轮廓来源（CDT / RGP / GT）会不会改变**图结构**？」
   这一点对 SAGEE 是致命的：它的边来自 `bot:adjacentZone`，而相邻关系是从
   **多边形共享边界**推出来的。若某套轮廓之间贴合不好，邻接会大幅丢失，
   SAGEE 就退化成"每个房间孤立地看自己的面积和门数"。
   所以同时报**孤立空间数**（邻接为 0 的空间）—— 它是轮廓质量的直接指标。

用法::

    python scripts/inspect_space_sets.py output/holdout_thr000 output/jsonld_rdp
    python scripts/inspect_space_sets.py output/gtchk output/rgpchk --limit 20
"""
from __future__ import annotations

import argparse
import json
import os


HEADERS = [
    ('spaces', '空间', '%d'),
    ('elements', '构件', '%d'),
    ('adj_undirected', '邻接边', '%d'),
    ('isolated', '孤立空间', '%d'),
]


def counts(path: str) -> dict:
    """统计一张图的 空间数 / 构件数 / 邻接边数 / 孤立空间数。"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    n_space = n_elem = adj_refs = n_isolated = 0
    for node in data.get('@graph') or []:
        if not isinstance(node, dict):
            continue
        types = node.get('@type') or []
        types = [types] if isinstance(types, str) else list(types)
        if 'bot:Space' in types:
            n_space += 1
            adj = node.get('bot:adjacentZone') or []
            adj = adj if isinstance(adj, list) else [adj]
            adj = [a for a in adj if a]
            adj_refs += len(adj)
            if not adj:
                n_isolated += 1
        elif 'bot:Element' in types:
            n_elem += 1

    return {
        'spaces': n_space,
        'elements': n_elem,
        # 邻接是双向记录的，所以无向边数 = 引用数 / 2
        'adj_undirected': adj_refs // 2,
        'isolated': n_isolated,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='对比各产物目录的空间/构件/邻接')
    ap.add_argument('dirs', nargs='+', help='一个或多个产物目录')
    ap.add_argument('--limit', type=int, default=12, help='最多打印多少张图纸（默认 12）')
    args = ap.parse_args()

    dirs = args.dirs
    labels = [os.path.basename(d.rstrip('/\\')) for d in dirs]
    names = sorted(n for n in os.listdir(dirs[0])
                   if n.endswith('.jsonld') and not n.endswith('_raw.jsonld'))[:args.limit]

    totals = {d: {k: 0 for k, _, _ in HEADERS} for d in dirs}
    per_drawing = []
    for name in names:
        row, ok = {}, True
        for d in dirs:
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                ok = False
                continue
            row[d] = counts(p)
        if ok:
            per_drawing.append((name, row))
            for d in dirs:
                for k, _, _ in HEADERS:
                    totals[d][k] += row[d][k]

    if not per_drawing:
        print('[!!] 这些目录里没有同名图纸')
        return 1

    print('=' * 78)
    print('按 %d 张同名图纸汇总' % len(per_drawing))
    print('-' * 78)
    print('%-18s %s' % ('指标', ''.join('%22s' % x for x in labels)))
    for key, label, fmt in HEADERS:
        print('%-18s %s' % (label, ''.join('%22s' % (fmt % totals[d][key]) for d in dirs)))

    for key, title in (('isolated', '孤立空间数 / 空间总数（越高说明轮廓之间贴合越差）'),
                       ('adj_undirected', '邻接边数')):
        print('\n--- 逐张图纸：%s ---' % title)
        print('%-26s %s' % ('图纸', ''.join('%20s' % x for x in labels)))
        for name, row in per_drawing:
            cells = []
            for d in dirs:
                if d not in row:
                    cells.append('%20s' % '-')
                elif key == 'isolated':
                    cells.append('%20s' % ('%d / %d' % (row[d]['isolated'], row[d]['spaces'])))
                else:
                    cells.append('%20s' % row[d][key])
            print('%-26s %s' % (name[:25], ''.join(cells)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
