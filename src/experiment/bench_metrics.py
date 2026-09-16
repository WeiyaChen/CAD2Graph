"""基准评估的共用核心：空间对齐、投票、混淆矩阵、指标。

为什么放在 ``src/`` 而不是 ``scripts/``
--------------------------------------
这套逻辑现在有三个使用者：命令行评估脚本、网页端汇总、以及以后的回归测试。
放在 ``scripts/`` 里的话网页端（``src/web_ui_server.py``）就 import 不到，只能复制
一份 —— 而"同一口径算了两次、两处不一致"是评估类代码最容易出的错。
所以把**唯一实现**放这里，脚本只做 CLI 与排版。

两个任务的评估单位
------------------
* **任务 1 · 空间轮廓提取**：直接比多边形。用 :func:`contour_metrics`。
* **任务 2 · 空间类型识别**：以 **GT 空间**为统计单位 —— 把落在同一个 GT 空间上的
  多块系统空间按多数票合并成一个预测，再判对错。用 :func:`type_confusion`。

  为什么不用系统空间做单位：不同分类器产出的系统空间数不一样（`TextMatching` 会把
  复合空间切开，`SAGEE` 是单标签、不切），按系统空间统计时各方法**分母不同**，
  不能直接比。以 GT 空间为单位则分母恒等于 GT 空间数，切分与否都不影响可比性。
  （实测见 `docs/sagee_baseline_plan.md` §6.5 / §6.6。）

⚠️ 弃权（未判定）**必须算漏检**。把弃权列从 F1 分母里去掉会让 macro-F1 从 0.46
虚高到 0.72（+56%），等于奖励"什么都不说"的分类器。
"""
from __future__ import annotations

import collections
import json
import os
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import Polygon

from src.spatial.sagee.graph import _as_polygon

__all__ = [
    'ABSTAIN', 'OUTSIDE',
    'drawing_key', 'load_gt', 'load_system_spaces',
    'match_to_gt', 'vote', 'col_of',
    'contour_metrics', 'type_confusion', 'metrics_from_cm',
]

#: 预测为空（置信度不足被丢弃 / 未判定）
ABSTAIN = '<未判定>'
#: 预测了一个不在当前标签空间内的类别
OUTSIDE = '<空间外>'


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
_KEY_RE = re.compile(r'^(?P<family>[0-9]+suite|sample|sample_Annotated)(?:_annotated|_Annotated)?'
                     r'\s*\((?P<idx>\d+)\)')


def drawing_key(path: str) -> str:
    """把系统/GT 文件名归一成同一个图纸键。

    系统侧是 ``2suite (1)_raw.jsonld``，GT 侧是 ``2suite_annotated (1)_gt.jsonld``，
    两者必须映射到同一个键（``2suite#1``）才能配对。``sample`` 没有编号，原样返回。

    （本函数原本是 ``scripts/export_sagee_dataset._key``；训练集导出与评估必须用
    同一个映射，否则训练/评估会配错图纸，所以提到共用位置。）
    """
    base = os.path.basename(path)
    m = _KEY_RE.match(base)
    if m:
        return '%s#%s' % (m.group('family'), m.group('idx'))
    return re.sub(r'(_raw|_gt)?\.jsonld$', '', base)


def _node_types(node: Mapping) -> list[str]:
    types = node.get('@type') or []
    return [types] if isinstance(types, str) else list(types)


def _iter_graph(path: str) -> Iterable[dict]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    for node in data.get('@graph') or []:
        if isinstance(node, dict):
            yield node


def _wkt_of(node: Mapping) -> str | None:
    wkt = node.get('geo:asWKT')
    if isinstance(wkt, dict):
        wkt = wkt.get('@value')
    return wkt or None


def load_gt(path: str) -> tuple[list[str], list[Polygon], list[str]]:
    """读 GT：返回 ``(id, 多边形, 类型名)``（类型取 ``bldg:`` 后面那段）。"""
    ids, polys, labels = [], [], []
    for node in _iter_graph(path):
        types = _node_types(node)
        if 'bot:Space' not in types:
            continue
        label = next((t.split(':', 1)[1] for t in types if t.startswith('bldg:')), None)
        if label is None:
            continue
        wkt = _wkt_of(node)
        if not wkt:
            continue
        try:
            from shapely.wkt import loads as wkt_loads
            poly = wkt_loads(wkt)
        except Exception:                                        # noqa: BLE001
            continue
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        ids.append(str(node.get('@id', '')))
        polys.append(poly)
        labels.append(label)
    return ids, polys, labels


def load_system_spaces(path: str) -> list[tuple[Any, list[str]]]:
    """读系统产物：返回 ``[(多边形, [bldg: 类型名...]), ...]``，忽略无几何的空间。"""
    out = []
    for node in _iter_graph(path):
        types = _node_types(node)
        if 'bot:Space' not in types:
            continue
        poly = _as_polygon(_wkt_of(node))
        if poly is None:
            continue
        out.append((poly, [t.split(':', 1)[1] for t in types if t.startswith('bldg:')]))
    return out


# --------------------------------------------------------------------------- #
# 对齐
# --------------------------------------------------------------------------- #
def match_to_gt(sys_polys: Sequence[Polygon | None], gt_polys: Sequence[Polygon],
                min_iou: float = 0.3) -> list[int]:
    """每个系统空间 → GT 下标（``-1`` = 没配上）。

    先取 IoU 最大者，不足 ``min_iou`` 时退回"系统空间质心落在哪个 GT 内"。
    与 ``scripts/export_sagee_dataset.match_labels`` 同一套逻辑 —— 训练集导出和
    评估必须同源，否则报出来的数字对不上训练时的数据分布。
    """
    out = [-1] * len(sys_polys)
    for i, poly in enumerate(sys_polys):
        if poly is None or poly.is_empty:
            continue
        best_j, best_iou = -1, 0.0
        for j, gpoly in enumerate(gt_polys):
            try:
                inter = poly.intersection(gpoly).area
                if inter <= 0:
                    continue
                iou = inter / (poly.area + gpoly.area - inter)
            except Exception:                                    # noqa: BLE001
                continue
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= min_iou:
            out[i] = best_j
            continue
        try:
            c = poly.centroid
        except Exception:                                        # noqa: BLE001
            continue
        for j, gpoly in enumerate(gt_polys):
            try:
                if gpoly.contains(c):
                    out[i] = j
                    break
            except Exception:                                    # noqa: BLE001
                continue
    return out


def vote(preds: Sequence[str]) -> str:
    """多数票合并同一 GT 空间上的多块系统空间；全弃权则弃权。"""
    real = [p for p in preds if p != ABSTAIN]
    if not real:
        return ABSTAIN
    cnt = collections.Counter(real)
    top = max(cnt.values())
    return sorted(c for c, v in cnt.items() if v == top)[0]


# --------------------------------------------------------------------------- #
# 任务 1：轮廓质量（直接比多边形，不涉及类型）
# --------------------------------------------------------------------------- #
def contour_metrics(sys_polys: Sequence[Polygon | None], gt_polys: Sequence[Polygon],
                    min_iou: float = 0.3, one_to_one_iou: float = 0.5) -> dict:
    """轮廓质量指标。

    * ``coverage``   —— 被至少一块系统轮廓覆盖到的 GT 空间比例（IoU≥``min_iou``
      或质心包含）。**这是轮廓漏检的直接度量。**
    * ``one_to_one`` —— 恰好被一块系统轮廓覆盖、且 IoU≥``one_to_one_iou`` 的 GT 空间
      比例。过合并/过切分都会压低它。
    * ``miou_matched`` —— 已配对的 GT 空间上的平均 IoU，衡量"形状准不准"。
    * ``count_ratio`` —— 系统轮廓数 / GT 空间数。1.0 最理想。
    * ``area_mae_m2`` —— 已配对空间上的平均面积绝对误差。
    """
    n_gt = len(gt_polys)
    n_sys = sum(1 for p in sys_polys if p is not None and not p.is_empty)
    res = {'n_sys': n_sys, 'n_gt': n_gt,
           'count_ratio': (n_sys / n_gt) if n_gt else 0.0,
           'coverage': 0.0, 'one_to_one': 0.0, 'miou_matched': 0.0,
           'area_mae_m2': 0.0, 'n_covered_gt': 0, 'n_matched': 0}
    if not n_gt or not n_sys:
        return res

    # 每个系统空间 -> 一个 GT 空间（含质心兜底）
    mapping = match_to_gt(sys_polys, gt_polys, min_iou)
    # 再算一遍精确 IoU，用于 miou / one-to-one 判定
    per_gt: dict[int, list[float]] = collections.defaultdict(list)
    for i, j in enumerate(mapping):
        if j < 0:
            continue
        poly = sys_polys[i]
        try:
            inter = poly.intersection(gt_polys[j]).area
            iou = inter / (poly.area + gt_polys[j].area - inter) if inter > 0 else 0.0
        except Exception:                                        # noqa: BLE001
            iou = 0.0
        per_gt[j].append(iou)

    covered = len(per_gt)
    one_to_one = sum(1 for ious in per_gt.values()
                     if len(ious) == 1 and ious[0] >= one_to_one_iou)
    ious_all = [x for ious in per_gt.values() for x in ious]
    area_err = []
    for i, j in enumerate(mapping):
        if j < 0:
            continue
        try:
            area_err.append(abs(sys_polys[i].area - gt_polys[j].area) / 1e6)
        except Exception:                                        # noqa: BLE001
            continue

    res.update({
        'n_covered_gt': covered,
        'n_matched': len(ious_all),
        'coverage': covered / n_gt,
        'one_to_one': one_to_one / n_gt,
        'miou_matched': float(np.mean(ious_all)) if ious_all else 0.0,
        'area_mae_m2': float(np.mean(area_err)) if area_err else 0.0,
    })
    return res


# --------------------------------------------------------------------------- #
# 任务 2：空间类型识别（以 GT 空间为统计单位）
# --------------------------------------------------------------------------- #
def col_of(pred: str, index_of: Mapping[str, int], n_class: int) -> int:
    """预测名 → 混淆矩阵列下标（未判定 / 空间外各占最后一列）。"""
    if pred in index_of:
        return index_of[pred]
    return n_class if pred == ABSTAIN else n_class + 1


def type_confusion(pairs: collections.Counter, classes: Sequence[str],
                   index_of: Mapping[str, int]) -> np.ndarray:
    """由 ``Counter{(gt, pred): n}`` 构造 ``(n_class, n_class+2)`` 混淆矩阵。"""
    n_class = len(classes)
    cm = np.zeros((n_class, n_class + 2), dtype=np.int64)
    for (g, p), c in pairs.items():
        if g in index_of:
            cm[index_of[g], col_of(p, index_of, n_class)] += c
    return cm


def metrics_from_cm(cm: np.ndarray) -> dict:
    """由混淆矩阵算 accuracy / coverage / macro-F1 / weighted-F1 / 逐类 F1。

    ⚠️ ``fn`` 取自**整行**（含弃权与空间外列）—— 弃权也是漏检。
    ``fp`` 只取自真实类别列（弃权/空间外不是一个"被预测的类别"，不产生 FP）。
    """
    n_class = cm.shape[0]
    block = cm[:, :n_class]
    tp = np.diag(block).astype(np.float64)
    fp = block.sum(0).astype(np.float64) - tp
    fn = cm.sum(1).astype(np.float64) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    support = cm.sum(1).astype(np.float64)
    present = support > 0
    counted = int(cm.sum())
    labeled = int(block.sum())
    return {
        'n_gt': counted,
        'n_labeled': labeled,
        'n_abstain': counted - labeled,
        'accuracy': float(tp.sum() / max(1, counted)),
        'coverage': float(labeled / max(1, counted)),
        'macro_f1': float(f1[present].mean()) if present.any() else 0.0,
        'weighted_f1': float((f1 * support).sum() / max(1, support.sum())),
        'f1_by_class': [float(x) for x in f1],
        'support': [int(x) for x in support],
        'present': [bool(x) for x in present],
    }
