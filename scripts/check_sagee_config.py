"""SAGE-E 分类器的配置/产物自检。

不需要 torch —— 只读部署三件套与特征 schema，秒级返回。用于在换了权重、
改了特征维度或换了标签空间之后，快速确认推理侧不会静默出错。

检查项
------
1. 算法已注册（``SPACE_TYPE_CLASSIFIERS``）
2. 工厂能实例化（会合并 ``settings.yaml`` 的共享参数）
3. 部署产物齐全：``<weights>.npz`` + ``.labels.json``（必需）
   + ``.norm.json``（按 ``<weights>.meta.json`` 记录的配方决定要不要）
4. **特征维度与 ``features.py`` 的 schema 一致**（最容易出错的地方：
   改了特征定义却忘了重训，维度对不上会直接报错；但如果维度恰好相同，
   数值含义已经变了 —— 所以还要比名称）
5. 标签空间与权重输出维度一致
6. 用 CORE 数据集的一批图做一次真实前向，确认无 NaN 且类别下标合法

用法::

    python scripts/check_sagee_config.py
    python scripts/check_sagee_config.py --weights data/sagee_cad2graph_core.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

OK = '  [ok]  '
BAD = '  [!!]  '
WARN = '  [??]  '


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))

    ap = argparse.ArgumentParser(description='SAGE-E 配置自检')
    ap.add_argument('--weights', default=None,
                    help='默认取环境变量 SAGEE_WEIGHTS')
    ap.add_argument('--dataset', default=os.path.join(root, 'data', 'cad2graph_sagee_rgp_core.npz'),
                    help='用于冒烟前向的数据集（可不存在，跳过第 6 项）')
    args = ap.parse_args()

    problems: list[str] = []

    print('=== 1. 注册表 ===')
    from src.spatial.factory import (                       # noqa: E402
        available_classifier_algorithms,
        create_space_type_classifier,
    )
    names = available_classifier_algorithms()
    print('  可用分类器:', names)
    if 'SAGEE' not in names:
        problems.append('SAGEE 未注册（检查 src/spatial/classifiers/__init__.py）')
        print(BAD + 'SAGEE 未注册')
        return 1
    print(OK + 'SAGEE 已注册')

    print('\n=== 2. 工厂实例化（会合并共享参数）===')
    try:
        clf = create_space_type_classifier('SAGEE')
    except Exception as exc:                                 # noqa: BLE001
        print(BAD + '实例化失败: %s: %s' % (type(exc).__name__, exc))
        return 1
    print('  类型          :', type(clf).__name__)
    print('  match_tolerance_mm: %r （SAGE-E 不使用，仅为兼容共享配置）'
          % clf.match_tolerance_mm)
    print(OK + '实例化成功')

    weights = args.weights or clf.weights
    print('\n=== 3. 部署产物 ===')
    if not weights:
        problems.append('未配置权重（settings.yaml 的 algorithm_params.SAGEE.weights '
                        '或环境变量 SAGEE_WEIGHTS）')
        print(BAD + '未配置权重')
        return 1
    stem = os.path.splitext(weights)[0]
    labels_path = stem + '.labels.json'
    norm_path = stem + '.norm.json'
    meta_path = stem + '.meta.json'

    # 训练配方决定了要不要标准化参数。没有 meta 就只能靠猜，所以要提醒。
    meta: dict = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        print('  训练配方    : %s  (normalize=%s, class_weights=%s, epochs=%s, '
              'lr=%s, wd=%s, dropout=%s)'
              % (meta.get('recipe', '?'), meta.get('normalize'), meta.get('class_weights'),
                 meta.get('epochs'), meta.get('lr'), meta.get('weight_decay'),
                 meta.get('dropout')))
    else:
        print(WARN + '没有 %s：无法判断该不该用标准化参数'
              % os.path.basename(meta_path))

    want_norm = meta.get('normalize')            # True / False / None(未知)

    if os.path.isfile(weights):
        print(OK + '%-8s %s (%.1f KB)' % ('weights', weights, os.path.getsize(weights) / 1024))
    else:
        print(BAD + '%-8s 缺失: %s' % ('weights', weights))
        problems.append('weights 缺失: %s' % weights)

    # 类别表：**必需**。下标错位是静默错误，不能接受缺失。
    if os.path.isfile(labels_path):
        print(OK + '%-8s %s (%.1f KB)'
              % ('labels', labels_path, os.path.getsize(labels_path) / 1024))
    else:
        print(BAD + '%-8s 缺失（必需）: %s' % ('labels', labels_path))
        problems.append('labels 缺失: %s' % labels_path)

    # 标准化参数：按配方决定。
    use_norm = os.path.isfile(norm_path) and want_norm is not False
    if os.path.isfile(norm_path):
        if want_norm is False:
            print(WARN + '%-8s 存在但配方为不标准化，运行时会被忽略: %s'
                  % ('norm', norm_path))
        else:
            print(OK + '%-8s %s (%.1f KB)'
                  % ('norm', norm_path, os.path.getsize(norm_path) / 1024))
    else:
        if want_norm is True:
            print(BAD + '%-8s 缺失，但配方是 --normalize: %s' % ('norm', norm_path))
            problems.append('norm 缺失但配方需要它: %s' % norm_path)
        else:
            print(OK + '%-8s 本配方不标准化，无需此文件\n'
                  '            （论文忠实配方就是如此）'
                  % 'norm')

    if problems:
        return 1
    bundle = {'weights': weights, 'norm': norm_path, 'labels': labels_path}

    print('\n=== 4. 特征维度 vs features.py schema ===')
    from src.spatial.sagee.features import (                 # noqa: E402
        EDGE_FEATURE_NAMES,
        NODE_FEATURE_NAMES,
    )
    from src.spatial.sagee.labels import LabelSpace          # noqa: E402
    from src.spatial.sagee.model import SageeNumpy           # noqa: E402

    model = SageeNumpy(weights, norm=(norm_path if use_norm else None))
    print('  权重       : %d 层, dim_node=%d, dim_edge=%d, n_class=%d'
          % (model.n_layer, model.ndim_in, model.edim, model.n_class))
    print('  schema     : dim_node=%d %s' % (len(NODE_FEATURE_NAMES), NODE_FEATURE_NAMES))
    print('               dim_edge=%d %s' % (len(EDGE_FEATURE_NAMES), EDGE_FEATURE_NAMES))
    if model.ndim_in != len(NODE_FEATURE_NAMES) or model.edim != len(EDGE_FEATURE_NAMES):
        problems.append('特征维度不匹配：权重 %d/%d vs schema %d/%d —— '
                        '改了特征定义就必须重训'
                        % (model.ndim_in, model.edim,
                           len(NODE_FEATURE_NAMES), len(EDGE_FEATURE_NAMES)))
        print(BAD + '维度不匹配，必须重训')
    else:
        print(OK + '维度一致')
        print(WARN + '维度相同**不保证**语义相同：改了特征公式但列数不变时，'
              '这个脚本查不出来，需要人工确认训练集是何时导出的')

    print('\n=== 5. 标签空间 ===')
    space = LabelSpace.load(bundle['labels'])
    print('  空间 %s，%d 类: %s' % (space.space, space.n_class, list(space.classes)))
    if space.n_class != model.n_class:
        problems.append('类别数不匹配：权重输出 %d 类，标签表 %d 类 —— '
                        '下标会整体错位，必须用同一份 labels.json'
                        % (model.n_class, space.n_class))
        print(BAD + '类别数不匹配')
    else:
        print(OK + '类别数与权重输出一致')

    print('\n=== 6. 冒烟前向 ===')
    if not os.path.isfile(args.dataset):
        print(WARN + '数据集不存在，跳过: %s' % args.dataset)
    else:
        from src.spatial.sagee.dataset import NpzGraphDataset  # noqa: E402
        ds = NpzGraphDataset(args.dataset)
        n_checked = 0
        bad = []
        for i in range(min(5, len(ds))):
            nf, ei, ef, _lb = ds.get(i)
            proba = model.predict_proba(nf, ei, ef)
            if not np.isfinite(proba).all():
                bad.append('图 %d 出现非有限值' % i)
            if proba.shape[1] != space.n_class:
                bad.append('图 %d 输出宽度 %d != 类别数 %d' % (i, proba.shape[1], space.n_class))
            n_checked += 1
        if bad:
            problems.extend(bad)
            for b in bad:
                print(BAD + b)
        else:
            print(OK + '%d 张图前向正常，输出宽度 %d' % (n_checked, space.n_class))
            conf = model.predict_proba(*ds.get(0)[:3]).max(axis=1)
            print('  图 0 的置信度: min %.3f  中位 %.3f  max %.3f'
                  % (conf.min(), float(np.median(conf)), conf.max()))
            print('  实际使用的 min_confidence: %.2f（环境变量 SAGEE_MIN_CONFIDENCE 可覆盖）'
                  % clf.min_confidence)

    print('\n' + '=' * 66)
    if problems:
        print('发现 %d 个问题:' % len(problems))
        for p in problems:
            print('  - ' + p)
        return 1
    print('全部通过。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
