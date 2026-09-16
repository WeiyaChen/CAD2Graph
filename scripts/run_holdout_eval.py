"""留出法在线评估：让每张图纸都被**没见过它**的模型预测。

为什么需要这个
--------------
``output/jsonld_sagee/`` 是**部署模型**（在全部 39 张图上训练过）的推理结果，
拿它去评估等于评估训练集本身 —— 数字虚高且不可比（实测 in-sample 0.717 vs
真正的样本外 0.55）。

本脚本用 ``train_sagee.py --cv K --emit-folds <dir>`` 导出的每折模型，只对
**该折留出的图纸**跑在线流水线。因为 K 折正好覆盖全部图纸，所以所有图纸都恰好
被一个「没见过它」的模型预测过一次，拼起来就是一份**完整的样本外在线结果**，
且与 TextMatching / LLMMultiStage 的零样本结果口径一致。

用法::

    # 1) 先导出每折模型（torch 环境）
    python scripts/train_sagee.py --dataset data/cad2graph_sagee_rgp_core.npz \
        --cv 5 --epochs 200 --select final \
        --labels-from data/cad2graph_sagee_rgp_core.labels.json \
        --emit-folds data/holdout_core

    # 2) 再跑留出评估（cadruler 环境，需要 shapely）
    python scripts/run_holdout_eval.py --folds data/holdout_core \
        --thresholds 0.0 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#: ``2suite#1`` -> ``2suite (1).svg``
_KEY_RE = re.compile(r'^(?P<family>.+?)#(?P<idx>\d+)$')


def key_to_svg(key: str) -> str | None:
    m = _KEY_RE.match(key)
    if not m:
        return None
    return '%s (%s).svg' % (m.group('family'), m.group('idx'))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='留出法在线评估驱动')
    ap.add_argument('--folds', default=os.path.join(root, 'data', 'holdout_core'))
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.5],
                    help='要扫的 min_confidence；每个阈值一份独立产物目录')
    ap.add_argument('--out-root', default=os.path.join(root, 'output'))
    ap.add_argument('--out-name', default=None,
                    help='固定产物目录名（仅一个阈值时有意义）。用于把结果直接写进基准矩阵，'
                         '例如 --out-root output/bench --out-name type_SAGEE')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--contour-algo', default='RGP')
    ap.add_argument('--only-fold', type=int, default=None, help='只跑某一折（调试用）')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.out_name and len(args.thresholds) != 1:
        print('[!!] --out-name 只在单个阈值下有意义（现在有 %d 个阈值）' % len(args.thresholds))
        return 2

    with open(os.path.join(args.folds, 'folds.json'), encoding='utf-8') as f:
        manifest = json.load(f)

    folds = manifest['folds']
    if args.only_fold:
        folds = [f for f in folds if f['fold'] == args.only_fold]

    total = sum(len(f['test_keys']) for f in folds) * len(args.thresholds)
    print('折数 %d，待跑图纸次数 %d（%d 折 × %d 个阈值）'
          % (len(folds), total, len(folds), len(args.thresholds)))

    results: dict[float, dict] = {}
    for thr in args.thresholds:
        tag = 'thr%03d' % round(thr * 100)
        out_dir = os.path.join(args.out_root, args.out_name or ('holdout_%s' % tag))
        os.makedirs(out_dir, exist_ok=True)
        results[thr] = {'dir': out_dir, 'runs': [], 'failed': []}

        print('\n===== min_confidence = %.2f -> %s =====' % (thr, out_dir))
        for fold in folds:
            weights = fold['weights']
            stem = os.path.splitext(weights)[0]
            env = dict(os.environ)
            env['PYTHONIOENCODING'] = 'utf-8'
            env['SAGEE_WEIGHTS'] = weights
            # 只在文件确实存在时才指定 SAGEE_NORM：论文忠实配方（不标准化）下没有
            # 这个文件，硬写进去会让分类器直接报错退出。
            norm_path = stem + '.norm.json'
            if os.path.isfile(norm_path):
                env['SAGEE_NORM'] = norm_path
            else:
                env.pop('SAGEE_NORM', None)
            env['SAGEE_LABELS'] = stem + '.labels.json'
            env['SAGEE_MIN_CONFIDENCE'] = str(thr)

            for key in fold['test_keys']:
                svg = key_to_svg(key)
                if svg is None:
                    results[thr]['failed'].append((key, 'key 无法解析成 SVG 名'))
                    continue
                if not os.path.isfile(os.path.join(root, 'input_data', 'svg', svg)):
                    results[thr]['failed'].append((key, 'SVG 不存在: %s' % svg))
                    continue
                cmd = [args.python, '-u', '-m', 'src.main', '--mode', 'SINGLE',
                       '--target-file', svg, '--output-dir', out_dir,
                       '--contour-algo', args.contour_algo,
                       '--classifier-algo', 'SAGEE']
                if args.dry_run:
                    print('  [dry] fold %d  %s' % (fold['fold'], svg))
                    continue
                t0 = time.time()
                proc = subprocess.run(cmd, cwd=root, env=env,
                                      capture_output=True, text=True,
                                      encoding='utf-8', errors='replace')
                ok = proc.returncode == 0 and os.path.isfile(
                    os.path.join(out_dir, key_to_svg(key).replace('.svg', '.jsonld')))
                if ok:
                    results[thr]['runs'].append(key)
                    print('  fold %d  %-22s ok  (%.1fs)' % (fold['fold'], svg, time.time() - t0))
                else:
                    results[thr]['failed'].append((key, proc.stderr[-400:] if proc.stderr else ''))
                    print('  fold %d  %-22s FAILED' % (fold['fold'], svg))

    print('\n===== 汇总 =====')
    for thr, info in results.items():
        print('min_confidence=%.2f  产物目录 %s  成功 %d  失败 %d'
              % (thr, info['dir'], len(info['runs']), len(info['failed'])))
        for key, why in info['failed'][:5]:
            print('    失败 %s: %s' % (key, why.splitlines()[-1] if why else ''))

    if not args.dry_run:
        with open(os.path.join(args.folds, 'holdout_runs.json'), 'w', encoding='utf-8') as f:
            json.dump({str(k): {'dir': v['dir'], 'n_ok': len(v['runs']),
                                'failed': v['failed']} for k, v in results.items()},
                      f, indent=2, ensure_ascii=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
