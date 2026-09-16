"""跑基准矩阵：把两个任务 × 各方法的产物统一落到 ``output/bench/`` 下。

为什么需要统一的目录
--------------------
之前每跑一次实验都新开一个目录（``jsonld`` / ``jsonld_rdp`` / ``jsonld_sagee`` /
``holdout_thr000``…），彼此的口径、图纸集合、算法组合都对不上，做汇总时只能靠人记。
本脚本把"一次运行 = 一个 (轮廓算法, 分类器) 组合"固定下来，一个组合一个目录，
于是评估脚本可以纯粹按目录名推断它是什么。

两个任务怎么落到这个矩阵上
--------------------------
用户定的口径是**两个独立任务、不做交叉**：

* **任务 1 · 空间轮廓提取** —— 比较 ``CDT / RGP / VecFloorSeg``（GT 是参照，不是方法）。
  这三次运行**一律用 ``NoOp`` 分类器**（不做任何类型识别）。原因有两条 ——

  1. 流水线必须有分类器。若落到 ``settings.yaml`` 的默认值（``LLMMultiStage``），
     一次纯轮廓实验会**白烧 41 次大模型调用**；
  2. 分类器会**改变轮廓本身**。富化阶段有 ACD 复合空间切分：一个空间若被判了多个
     类型就会被切开（实测同一张图在 TextMatching 下 42 个空间、单标签分类器下 36 个）。
     拿带分类器的产物去量轮廓，量到的是"分类器 + 提取器"的混合物。

  ⚠️ 任务 1 的**评估**读的也是 ``<base>_raw.jsonld``（富化**之前**写出的那份图，
  只有 ``bot:Space`` 和几何）—— 它在任何分类器下都**语义等价**，这才是"提取出来的轮廓"。

  （实测：同一张图在 ``SAGEE`` / ``LLMMultiStage`` / ``TextMatching`` 三个目录下的
  ``_raw.jsonld`` 大小、``@id`` 集合、空间数、几何全部一致，仅 ``bot:containsElement``
  等**无序关系列表的元素顺序**不同 —— 这是 README「Notes」里已记录的固有非确定性，
  所以不要按字节比对，要比就比集合。）

* **任务 2 · 空间类型识别** —— 比较 ``LLMMultiStage / TextMatching / SAGEE``。
  **必须在同一套轮廓上跑**，否则差异里混进了轮廓误差，不是有效 ablation。
  所以这三次运行**全部用 ``--contour-algo GT``**。

于是 3 + 3 = **6 次运行**，两个任务的产物互不共用。

关于 sample
-----------
``input_data/svg/`` 里的 ``sample.svg`` 只是为了跑通流程的样例，用户明确要求
benchmark 不计入。``--mode BATCH`` 无法按文件名过滤，所以**这里仍然会跑它**，
统一在**评估阶段排除**（见 ``scripts/evaluate_benchmark.py --exclude sample``）。

用法::

    python scripts/run_benchmark.py --dry-run              # 只看要跑什么
    python scripts/run_benchmark.py                        # 跑所有不完整的
    python scripts/run_benchmark.py --run GT_SAGEE         # 只跑一个
    python scripts/run_benchmark.py --force --run RGP_TextMatching
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
BENCH_DIR = os.path.join(ROOT, 'output', 'bench')

#: 每个任务自己的候选方法
CONTOUR_ALGOS = ('CDT', 'RGP', 'VecFloorSeg')
CLASSIFIERS = ('LLMMultiStage', 'SAGEE', 'TextMatching')

#: 端到端 run 的执行顺序 —— 先把便宜、出结果快的填满矩阵：
#:   1) CDT / RGP 的 SAGEE + TextMatching  秒级，6 格里先有 4 格
#:   2) VecFloorSeg 的那两个              要起它自己 conda 环境的子进程，慢
#:   3) LLMMultiStage 的三个              最慢 —— 实测单次 LLM 往返 10~40 秒，
#:                                        一张图要十几次调用，一个 run 是小时级
E2E_PAIRS: tuple[tuple[str, str], ...] = (
    tuple((c, k) for k in ('SAGEE', 'TextMatching') for c in ('CDT', 'RGP'))
    + tuple((c, k) for k in ('SAGEE', 'TextMatching') for c in ('VecFloorSeg',))
    + tuple((c, 'LLMMultiStage') for c in CONTOUR_ALGOS)
)

#: (run 名, 轮廓算法, 分类器, 它服务于哪些任务)
#: 目录名约定：
#:   ``contour_<轮廓算法>``   任务 1（分类器固定 ``NoOp``）
#:   ``type_<分类器>``        任务 2（轮廓固定 ``GT``）
#:   ``e2e_<轮廓算法>_<分类器>``  端到端：任务 2 **跑在任务 1 的轮廓上**
RUNS: list[tuple[str, str, str, tuple[str, ...]]] = [
    ('contour_CDT',         'CDT',         'NoOp',          ('contour',)),
    ('contour_RGP',         'RGP',         'NoOp',          ('contour',)),
    ('contour_VecFloorSeg', 'VecFloorSeg', 'NoOp',          ('contour',)),
    ('type_TextMatching',   'GT',          'TextMatching',  ('type',)),
    ('type_SAGEE',          'GT',          'SAGEE',         ('type',)),
    ('type_LLMMultiStage',  'GT',          'LLMMultiStage', ('type',)),
] + [
    ('e2e_%s_%s' % (contour, clf), contour, clf, ('e2e',))
    for contour, clf in E2E_PAIRS
]

TASK_LABEL = {'contour': '任务1·空间轮廓提取', 'type': '任务2·空间类型识别（GT轮廓）',
              'e2e': '端到端·任务1×任务2'}


def benchmark_keys(gt_dir: str, exclude: tuple[str, ...]) -> set[str]:
    """**基准集**：有非空 GT、且不被排除的图纸键。

    实测 ``output/gt/6suite_annotated (5)_gt.jsonld`` 是**空文件**（``"@graph": []``）
    —— 那张图的 GT 标注从来没生成过。若把"图纸数 == 41"当完成判据，这类图会让 run
    永远判定为未完成、反复重跑。所以完成度以**基准集**为准：

        41 张 dxf → 排除 sample（用户要求） → 排除 6suite (5)（无 GT） → **39 张**
    """
    sys.path.insert(0, ROOT)
    from src.experiment.bench_metrics import drawing_key, load_gt      # noqa: PLC0415

    keys: set[str] = set()
    for p in glob.glob(os.path.join(gt_dir, '*.jsonld')):
        k = drawing_key(p)
        if any(k.lower().startswith(x.lower()) for x in exclude):
            continue
        try:
            _ids, polys, _labels = load_gt(p)
        except Exception:                                              # noqa: BLE001
            continue
        if polys:
            keys.add(k)
    return keys


def produced_keys(out_dir: str) -> set[str]:
    """该目录里已产出的图纸键（不含 ``_raw.jsonld``）。"""
    sys.path.insert(0, ROOT)
    from src.experiment.bench_metrics import drawing_key               # noqa: PLC0415

    if not os.path.isdir(out_dir):
        return set()
    return {drawing_key(n) for n in os.listdir(out_dir)
            if n.endswith('.jsonld') and not n.endswith('_raw.jsonld')}


def run_dir(run_name: str) -> str:
    """某个 run 的产物目录。"""
    return os.path.join(BENCH_DIR, run_name)


def print_status(bench_dir: str, bench: set[str]) -> None:
    """列出 ``output/bench/`` 下每个目录对基准集的覆盖情况。

    （写在这里而不是临时敲 shell：PowerShell 的变量插值在这个环境里非常容易被打断，
    已经因此浪费过好几次统计。）
    """
    dirs = sorted(d for d in os.listdir(bench_dir)
                  if os.path.isdir(os.path.join(bench_dir, d))) \
        if os.path.isdir(bench_dir) else []
    if not dirs:
        print('  （%s 下还没有目录）' % os.path.relpath(bench_dir, ROOT))
        return
    print('%-26s %10s %10s  %s' % ('目录', '覆盖', '产物数', '缺失（最多显示 5 个）'))
    print('-' * 96)
    for name in dirs:
        d = os.path.join(bench_dir, name)
        have = produced_keys(d) & bench
        missing = sorted(bench - have)
        print('%-26s %5d/%-4d %10d  %s'
              % (name, len(have), len(bench), len(produced_keys(d)),
                 ', '.join(missing[:5]) + (' …' if len(missing) > 5 else '')))


def main() -> int:
    ap = argparse.ArgumentParser(description='跑基准矩阵（两个任务 × 各方法）')
    ap.add_argument('--run', action='append', default=None,
                    metavar='NAME', help='只跑指定 run（可重复）；默认全部')
    ap.add_argument('--task', choices=['contour', 'type', 'e2e'], default=None,
                    help='只跑某个任务下的 run')
    ap.add_argument('--gt-dir', default=os.path.join(ROOT, 'output', 'gt'))
    ap.add_argument('--target-dir', default=None,
                    help='传给 --mode BATCH 的输入目录（绝对路径或 input_data/svg 下的子目录名）。\n'
                         '用来把跑批精度控制到基准集：--mode BATCH 是按目录遍历的，\n'
                         '无法按文件名排除，直接跑会把 sample / 无 GT 的图也算进去，\n'
                         '对 LLMMultiStage 这种按图次计费的方法是真金白银的浪费。')
    ap.add_argument('--exclude', nargs='*', default=['sample'],
                    help='不纳入基准集的图纸键前缀（默认 sample）')
    ap.add_argument('--force', action='store_true', help='即使已完整也重跑')
    ap.add_argument('--dry-run', action='store_true', help='只打印计划，不执行')
    ap.add_argument('--status', action='store_true',
                    help='只列出 output/bench 下所有目录的覆盖情况，不跑任何东西')
    args = ap.parse_args()

    bench = benchmark_keys(args.gt_dir, tuple(args.exclude or ()))
    if args.status:
        print('基准目录: %s' % os.path.relpath(BENCH_DIR, ROOT))
        print('基准集  : %d 张图纸（已排除 %s）\n'
              % (len(bench), ', '.join(args.exclude or ['（无）'])))
        print_status(BENCH_DIR, bench)
        return 0

    selected = []
    for name, contour, clf, tasks in RUNS:
        if args.run and name not in args.run:
            continue
        if args.task and args.task not in tasks:
            continue
        selected.append((name, contour, clf, tasks))

    if not selected:
        print('[!!] 没有匹配的 run。可选：%s' % ', '.join(n for n, *_ in RUNS))
        return 2

    bench = benchmark_keys(args.gt_dir, tuple(args.exclude or ()))
    print('基准目录: %s' % os.path.relpath(BENCH_DIR, ROOT))
    print('基准集  : %d 张图纸（已排除 %s；GT 为空或缺 GT 的图不计入）\n'
          % (len(bench), ', '.join(args.exclude or ['（无）'])))
    print('%-26s %-11s %-14s %-24s %8s  %s'
          % ('run', '轮廓算法', '分类器', '服务于', '覆盖', '状态'))
    print('-' * 106)
    todo = []
    for name, contour, clf, tasks in selected:
        d = run_dir(name)
        have = produced_keys(d) & bench
        missing = bench - have
        done = not missing and not args.force
        state = '已完成，跳过' if done else ('重跑' if have else '待跑')
        if not done:
            todo.append((name, contour, clf, sorted(missing)))
        print('%-26s %-11s %-14s %-24s %8s  %s'
              % (name, contour, clf,
                 '/'.join(TASK_LABEL[t] for t in tasks),
                 '%d/%d' % (len(have), len(bench)), state))
        if missing and len(missing) <= 5:
            print('    缺: %s' % ', '.join(sorted(missing)))

    if not todo:
        print('\n全部已完成。')
        return 0
    if args.dry_run:
        print('\n--dry-run：不执行。')
        return 0

    print('\n开始执行 %d 个 run …' % len(todo))
    manifest_path = os.path.join(BENCH_DIR, 'manifest.json')
    manifest: dict = {'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
                      'benchmark_keys': sorted(bench),
                      'excluded_prefixes': list(args.exclude or []),
                      'runs': {}}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    manifest['benchmark_keys'] = sorted(bench)
    manifest.setdefault('runs', {})

    t_all = time.time()
    for name, contour, clf, _missing in todo:
        d = run_dir(name)
        os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, '-u', '-m', 'src.main', '--mode', 'BATCH',
               '--output-dir', os.path.relpath(d, ROOT),
               '--contour-algo', contour, '--classifier-algo', clf]
        if args.target_dir:
            cmd += ['--target-dir', args.target_dir]
        print('\n[%s] %s' % (name, ' '.join(cmd[3:])))
        t0 = time.time()
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        proc = subprocess.run(cmd, cwd=ROOT, env=env,
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace')
        dt = time.time() - t0
        have = produced_keys(d) & bench
        still = sorted(bench - have)
        manifest['runs'][name] = {
            'contour_algo': contour, 'classifier_algo': clf,
            'output_dir': os.path.relpath(d, ROOT),
            'returncode': proc.returncode,
            'n_produced': len(produced_keys(d)),
            'n_bench_covered': len(have), 'n_bench': len(bench),
            'missing': still,
            'seconds': round(dt, 1),
            'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        print('   rc=%d  覆盖 %d/%d  缺 %d   %.1fs'
              % (proc.returncode, len(have), len(bench), len(still), dt))
        if still:
            print('   缺: %s' % ', '.join(still[:8]))
        if proc.returncode != 0:
            print('   [!!] 非零退出，尾部输出：')
            print('   ' + '\n   '.join((proc.stderr or proc.stdout or '').strip().splitlines()[-8:]))

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print('\n合计 %.1fs；清单已写出: %s' % (time.time() - t_all, manifest_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
