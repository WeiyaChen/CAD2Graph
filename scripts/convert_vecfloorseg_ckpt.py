"""把 VecFloorSeg 的 ckpt 转换成「本地代码能加载」的形式（剥掉多余的 ``module.`` 前缀）。

背景
----
服务器上下载的 ``best1.ckpt`` 和本地 ``0.ckpt`` 相比：

* 851 个共同张量，形状全同，参数总量逐位一致（42,609,147）—— **架构相同**；
* 但 ``best1.ckpt`` 有 80 个键多一层 ``module.``，例如

      A(远程): model.mp.layer0.layer.edgeStage.model.conv.mlp.norms.0.module.weight
      B(本地): model.mp.layer0.layer.edgeStage.model.conv.mlp.norms.0.weight

  **只有这 80 个（全是 norm 相关）**，其余键名完全一致 —— 所以**不是**整模型 DDP 包装，
  而是服务器那版的代码把 norm 多包了一层。

为什么必须在本地转换
--------------------
``graphgym/checkpoint.py`` 用的是 ``model.load_state_dict(ckpt['model_state'])``，
**默认 strict=True**。键集合不匹配会直接抛 unexpected/missing keys。

而我们的适配器（``src/spatial/vecfloorseg/runner.py::_stage_checkpoint``）是在
**cadruler 环境**里跑的 —— 那个环境**没有 torch**，所以不能在那里剥前缀。
因此做成一次性的离线转换：转好一份，让 ``VECFLOORSEG_CKPT`` 指向它。

用法::

    python scripts/convert_vecfloorseg_ckpt.py <src.ckpt> <dst.ckpt> \\
        [--reference <local.ckpt>]

加了 ``--reference`` 会顺带核对：剥完前缀后键集合是否与本地权重**完全一致**。
一致才说明「只是前缀差异」，不一致就说明代码版本差异更大，需要另查。
"""
from __future__ import annotations

import argparse
import os
import shutil

PREFIX = 'module.'


def _strip(key: str) -> str:
    """去掉多余的 ``module`` 段。

    实测远程权重的多余段在**中间**而不是开头：

        model.mp...norms.0.module.weight   ->   model.mp...norms.0.weight

    （开头那种是整模型 DDP 包装，也一并处理，以防万一。）
    """
    if key.startswith(PREFIX):
        return key[len(PREFIX):]
    return key.replace('.' + PREFIX, '.')


def _has_prefix(key: str) -> bool:
    return key.startswith(PREFIX) or ('.' + PREFIX) in key


def main() -> int:
    ap = argparse.ArgumentParser(description='剥掉 VecFloorSeg ckpt 的多余 module. 前缀')
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--reference', default=None,
                    help='本地权重，用于核对剥完前缀后键集合是否一致')
    ap.add_argument('--force', action='store_true', help='目标已存在时也覆盖')
    args = ap.parse_args()

    import torch

    if not os.path.isfile(args.src):
        print('[!!] 源文件不存在: %s' % args.src)
        return 1
    if os.path.exists(args.dst) and not args.force:
        print('目标已存在（用 --force 覆盖）: %s' % args.dst)
        return 0

    print('读取 %s (%.1f MB) …' % (args.src, os.path.getsize(args.src) / 1e6))
    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state' not in ckpt:
        print('[!!] 不是 GraphGym 格式（缺 model_state）')
        return 1

    sd = ckpt['model_state']
    hits = [k for k in sd if _has_prefix(k)]
    print('model_state: %d 个张量，其中 %d 个带多余 %r 段' % (len(sd), len(hits), PREFIX))
    if hits:
        print('  例：')
        for k in sorted(hits)[:3]:
            print('    %s\n      -> %s' % (k, _strip(k)))
    if not hits:
        print('没有需要剥的前缀，直接复制。')
        os.makedirs(os.path.dirname(os.path.abspath(args.dst)) or '.', exist_ok=True)
        shutil.copy2(args.src, args.dst)
        return 0

    stripped = {_strip(k): v for k, v in sd.items()}
    ckpt['model_state'] = stripped
    if len(stripped) != len(sd):
        print('[!!] 剥完前缀后张量数变了（%d -> %d）—— 存在键冲突，不能这样转'
              % (len(sd), len(stripped)))
        return 1
    print('剥掉前缀后仍是 %d 个张量，无冲突。' % len(stripped))

    if args.reference:
        ref = torch.load(args.reference, map_location='cpu', weights_only=False)
        ref_sd = ref.get('model_state') or {}
        only_new = set(stripped) - set(ref_sd)
        only_ref = set(ref_sd) - set(stripped)
        print('与参考权重 %s 核对：' % os.path.basename(args.reference))
        print('  参考张量 %d 个；仅转换后有 %d 个，仅参考有 %d 个'
              % (len(ref_sd), len(only_new), len(only_ref)))
        if not only_new and not only_ref:
            print('  [ok] 键集合**完全一致** —— 确认只是前缀差异，可以放心用。')
        else:
            print('  [!!] 键集合仍不一致 —— 不只是前缀问题，服务器代码版本可能差更多。')
            for tag, ks in (('仅转换后', only_new), ('仅参考', only_ref)):
                for k in sorted(ks)[:10]:
                    print('       %s: %s' % (tag, k))

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)) or '.', exist_ok=True)
    torch.save(ckpt, args.dst)
    print('已写出 %s (%.1f MB)' % (args.dst, os.path.getsize(args.dst) / 1e6))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
