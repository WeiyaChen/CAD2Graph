"""读 VecFloorSeg 的 ``.ckpt`` 元信息，判断它是哪一次训练的产物。

为什么需要这个
--------------
``third_party/VecFloorSeg/results/CUBI/0/`` 里同时躺着 ``config.yaml``、
``logging.log`` 和 ``ckpt/*.ckpt``。但那份 config 日志可能是**别的** ckpt 留下的
（实测它写着 ``max_epoch: 1`` / ``out_dir: results_smoke\\CUBI``，即一次烟雾测试）。
一个 ckpt 是否"训练好了"，不能靠旁边的文件猜，必须读 ckpt 自己。

格式（GraphGym，非 Lightning）
------------------------------
``torch.save({model_state, optimizer_state, scheduler_state})`` —— **内部不含 epoch**。
轮次只编码在**文件名**里：``get_ckpt_path(epoch, prefix) = {prefix}{epoch}.ckpt``。
所以 ``0.ckpt`` = epoch 0，``best1.ckpt`` = ``best`` 前缀、epoch 1。

用法::

    python scripts/inspect_vecfloorseg_ckpt.py <a.ckpt> [<b.ckpt> ...]
    # 传两个时额外做权重比对：完全相同 -> 同一次训练；不同 -> 不同产物
"""
from __future__ import annotations

import argparse
import datetime
import os
import re

_BASENAME_RE = re.compile(r'^(?P<prefix>[A-Za-z_]*)(?P<epoch>\d+)\.ckpt$')


def _describe_name(path: str) -> None:
    base = os.path.basename(path)
    m = _BASENAME_RE.match(base)
    if not m:
        print('  文件名解析      : 不符合 {prefix}{epoch}.ckpt 约定')
        return
    prefix = m.group('prefix') or '(无)'
    epoch = int(m.group('epoch'))
    print('  文件名解析      : prefix=%s  epoch=%s' % (prefix, epoch))
    if prefix == 'best':
        print('                    -> GraphGym 存的「最佳轮次」权重；best1 = 最佳出现在第 1 轮')
    if epoch == 0 and prefix != 'best':
        print('                    -> 第 0 轮；若这次训练只跑了 1 轮就是烟雾测试产物')


def inspect(path: str) -> dict:
    import torch

    print('=' * 74)
    print('文件: %s' % path)
    if not os.path.isfile(path):
        print('  [!!] 不存在')
        return {}
    print('  大小 / 时间     : %.1f MB   %s'
          % (os.path.getsize(path) / 1e6,
             datetime.datetime.fromtimestamp(os.path.getmtime(path))
             .strftime('%Y-%m-%d %H:%M:%S')))
    _describe_name(path)

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        print('  [!!] 不是 dict，而是 %s' % type(ckpt).__name__)
        return {}

    print('  顶层键          : %s' % sorted(ckpt.keys()))
    sd = None
    for key in ('model_state', 'state_dict'):
        if isinstance(ckpt.get(key), dict):
            sd = ckpt[key]
            print('  权重字典        : ckpt[%r]' % key)
            break
    if sd is None:
        print('  [!!] 找不到权重字典（既无 model_state 也无 state_dict）')
        return ckpt

    n = sum(int(v.numel()) for v in sd.values() if hasattr(v, 'numel'))
    print('  张量 / 参数量   : %d 个 / %s' % (len(sd), format(n, ',')))
    for k in ('optimizer_state', 'scheduler_state'):
        print('  %-15s : %s' % (k, '有' if ckpt.get(k) else '无'))
    return {'ckpt': ckpt, 'sd': sd, 'path': path}


def compare(all_a, all_b) -> None:
    import torch

    sa, sb = all_a['sd'], all_b['sd']
    keys_a, keys_b = set(sa), set(sb)
    print('=' * 74)
    print('权重比对')
    print('  A: %s' % all_a['path'])
    print('  B: %s' % all_b['path'])
    only_a, only_b = keys_a - keys_b, keys_b - keys_a
    if only_a or only_b:
        print('  [!!] 键集合不同：仅 A 有 %d 个，仅 B 有 %d 个' % (len(only_a), len(only_b)))
        for tag, ks in (('仅 A', only_a), ('仅 B', only_b)):
            print('     %s 示例:' % tag)
            for k in sorted(ks)[:12]:
                print('       %s' % k)
    common = sorted(keys_a & keys_b)
    n_diff = 0
    worst, worst_key = 0.0, ''
    for k in common:
        va, vb = sa[k], sb[k]
        if getattr(va, 'shape', None) != getattr(vb, 'shape', None):
            n_diff += 1
            continue
        if getattr(va, 'dtype', None) is not None and va.dtype.is_floating_point:
            d = float(torch.max(torch.abs(va.float() - vb.float()))) if va.numel() else 0.0
        else:
            d = 0.0 if torch.equal(va, vb) else 1.0
        if d > 0:
            n_diff += 1
        if d > worst:
            worst, worst_key = d, k
    print('  共同张量 %d 个，其中 %d 个不同' % (len(common), n_diff))
    if n_diff == 0 and not only_a and not only_b:
        print('  [结论] 两个 ckpt 权重**完全相同** —— 同一次训练的同一份产物（可能只是拷贝）')
    else:
        print('  [结论] 权重**不同**（最大差异 %.4g，出现在 %s）\n'
              '        —— 两者来自不同的训练/轮次，不能混用。' % (worst, worst_key))


def main() -> int:
    ap = argparse.ArgumentParser(description='读 / 比对 VecFloorSeg ckpt')
    ap.add_argument('paths', nargs='+')
    args = ap.parse_args()

    infos = []
    for p in args.paths:
        r = inspect(p)
        if r:
            infos.append(r)
    if len(infos) == 2:
        compare(infos[0], infos[1])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
