"""自检 VecFloorSeg baseline 的配置接线是否正确。

只做静态检查，**不跑推理**（秒级返回），用于在改完 ``settings.yaml`` / ``.env``
后确认：算法已注册、工厂能实例化、机器相关参数（解释器 / 权重 / 仓库根目录）
确实被解析到，以及可视化契约仍然满足。

用法::

    cadruler\\Scripts\\python.exe scripts\\check_vecfloorseg_config.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spatial.config import SpatialPipelineConfig            # noqa: E402
from src.spatial.factory import (                               # noqa: E402
    available_contour_algorithms,
    create_contour_extractor,
)

OK = '  [ok]  '
BAD = '  [!!]  '


def main() -> int:
    problems: list[str] = []

    print('=== 1. 注册表 ===')
    names = available_contour_algorithms()
    print('  可用算法:', names)
    if 'VECFLOORSEG' not in names:
        problems.append('VecFloorSeg 未注册（检查 src/spatial/contours/__init__.py 的导入）')
        print(BAD + 'VecFloorSeg 未注册')
    else:
        print(OK + 'VecFloorSeg 已注册')

    print('\n=== 2. settings.yaml 解析 ===')
    cfg = SpatialPipelineConfig.from_settings()
    print('  默认算法          :', cfg.contour_algorithm)
    groups = sorted(cfg.contour_algorithm_params)
    print('  algorithm_params  :', groups)
    if not any(g.upper() == 'VECFLOORSEG' for g in groups):
        problems.append('settings.yaml 缺少 spatial.contour.algorithm_params.VecFloorSeg')
        print(BAD + '缺少 VecFloorSeg 参数组；下面将只用默认值继续检查')
    else:
        print(OK + 'VecFloorSeg 参数组已配置')

    print('\n=== 3. 工厂实例化（显式指定 VecFloorSeg）===')
    try:
        ex = create_contour_extractor('VecFloorSeg', config=cfg)
    except Exception as exc:                                     # noqa: BLE001
        print(BAD + '实例化失败: %s: %s' % (type(exc).__name__, exc))
        return 1
    print('  类型      :', type(ex).__name__)
    print('  注册名    :', ex.name)
    print('  可视化后缀:', ex.visualization_suffix)
    print(OK + '实例化成功')

    print('\n=== 4. 机器相关路径（环境变量 / 配置） ===')
    for label, value, kind in (
        ('python_exe', ex.python_exe, 'file'),
        ('checkpoint', ex.checkpoint, 'file'),
        ('vf_root', ex.vf_root or '(未设置 -> third_party/VecFloorSeg)', 'dir'),
    ):
        exists = (
            os.path.isfile(value) if kind == 'file' and value
            else os.path.isdir(value) if kind == 'dir' and os.path.exists(value)
            else None
        )
        if exists is None:                    # vf_root 未设置属于正常回退
            print('  %-11s: %s' % (label, value))
        elif exists:
            print(OK + '%-11s: %s' % (label, value))
        else:
            print(BAD + '%-11s: %s  <-- 不存在' % (label, value))
            if kind == 'file':
                problems.append('%s 指向的文件不存在: %s' % (label, value))

    print('\n=== 5. 与机器无关的参数 ===')
    for key in ('device', 'canvas_px', 'margin_ratio', 'wall_thickness_mm',
                'simplify_tol_mm', 'timeout_s', 'keep_workdir', 'keep_labels'):
        print('  %-18s: %r' % (key, getattr(ex, key)))

    print('\n=== 6. 可视化契约（plot_floor_plan 解包 4 项）===')
    data = ex.get_visualization_data()
    print('  len =', len(data))
    if len(data) != 4:
        problems.append('get_visualization_data() 必须返回 4 项，实际 %d' % len(data))
        print(BAD + '返回 %d 项，plot_floor_plan 会报错' % len(data))
    else:
        print(OK + '恰好 4 项')

    print('\n=== 7. 共享过滤器参数（DEFAULT_CONTOUR_PARAMS 恒被合并）===')
    missing = [
        k for k in ('min_area_mm2', 'erode_mm', 'min_width_mm', 'min_compactness',
                    'min_solidity', 'virtual_blocker_dist_tol_mm',
                    'virtual_blocker_max_len_mm', 'virtual_blocker_angle_tol_deg')
        if k not in ex.shape_filter_params
    ]
    if missing:
        problems.append('shape_filter_params 缺少: %s' % ', '.join(missing))
        print(BAD + '缺少 %s' % ', '.join(missing))
    else:
        print(OK + '8 个共享参数齐全')
        print('   过滤阈值: min_area=%.0f mm²  erode=%.0f mm  min_width=%.0f mm'
              % (ex.shape_filter_params['min_area_mm2'],
                 ex.shape_filter_params['erode_mm'],
                 ex.shape_filter_params['min_width_mm']))

    print('\n=== 8. algorithm_params 在运行期切换算法时仍然生效 ===')
    print('  settings.yaml 的默认算法是 %r，但 CLI/Web UI 可以改用别的；'
          % cfg.contour_algorithm)
    print('  此时必须按 *最终* 算法名重新取参，否则该算法的专属参数会被静默忽略。')
    for algo in ('CDT', 'RGP', 'VecFloorSeg'):
        try:
            p = cfg.contour_params_for(algo)
        except Exception as exc:                                 # noqa: BLE001
            problems.append('contour_params_for(%r) 抛错: %s' % (algo, exc))
            print(BAD + '%-12s -> %s: %s' % (algo, type(exc).__name__, exc))
            continue
        group = next((g for g in cfg.contour_algorithm_params
                      if g.upper() == algo.upper()), None)
        expected = set(cfg.contour_algorithm_params.get(group, {})) if group else set()
        got = sorted(set(p) & expected)
        if expected and len(got) != len(expected):
            problems.append('%s 的 algorithm_params 未被合并（期望 %d 项，实得 %d 项）'
                            % (algo, len(expected), len(got)))
            print(BAD + '%-12s 合并 %d/%d 项' % (algo, len(got), len(expected)))
        elif expected:
            print(OK + '%-12s 专属参数 %d 项已合并' % (algo, len(got)))
        else:
            print('  %-12s 无 algorithm_params 分组（正常）' % algo)

    print('\n' + '=' * 62)
    if problems:
        print('发现 %d 个问题:' % len(problems))
        for p in problems:
            print('  - ' + p)
        return 1
    print('全部通过。可用 --contour-algo VecFloorSeg 运行流水线。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
