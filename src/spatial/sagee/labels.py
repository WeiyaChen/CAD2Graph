"""标签空间：长尾折叠 + **稳定**的类别索引。

为什么需要这一层
----------------
1. GT 里有 21 个空间类型，其中 ``SunRoom`` 只有 3 个样本、``Corridor`` 只有 6 个 ——
   这种量级既学不动、也会让 macro-F1 剧烈抖动。需要显式折叠规则，而不是"跳过算了"。
2. 模型的输出是**类别下标**。下标 → 类名的对应关系必须**跨进程、跨次训练完全一致**，
   否则训练出来的权重在推理时会整体错位。因此类别列表随权重一起落盘
   （见 :meth:`LabelSpace.save`），推理时**必须**从同一份文件读取，不能各自推断。
3. GT 的类型写在 ``@type`` 里（形如 ``["bot:Space", "bldg:Bedroom"]``），所以折叠发生在
   ``bldg:`` 之后的那一段名字上。

两个预设标签空间
----------------
``FULL``
    全部空间（含电梯井、楼梯间、设备间等）。只把样本数 < :data:`MIN_SAMPLES` 的类别
    折到语义最近的父类。默认。
``CORE``
    只保留**套内居住空间**，把公共/结构类整体排除。适合和"户型理解"类工作对比。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    'DEFAULT_SPACE',
    'MIN_SAMPLES',
    'LONG_TAIL_FOLDING',
    'CORRIDOR_LABELS',
    'CORRIDOR_TARGET',
    'LabelSpace',
    'fold_label',
    'folding_for',
    'resolve_classes',
]

#: 默认标签空间
DEFAULT_SPACE = 'FULL'
#: 样本数低于该值的类别会被折叠（见 :data:`LONG_TAIL_FOLDING`）
MIN_SAMPLES = 10

#: 长尾折叠规则：原始类型 → 语义最近的父类。
#: 依据是实测的 GT 分布（41 份标注图 / 1951 个空间）：
#: ``SunRoom`` 3 个、``Corridor`` 6 个、``VentilationRoom`` 10 个。
LONG_TAIL_FOLDING: dict[str, str] = {
    'SunRoom': 'Balcony',            # 阳光房 → 阳台（同为半室外延伸空间）
    'Corridor': 'SecondaryCorridor',  # 泛化的"走廊" → 次走廊
    'VentilationRoom': 'ElectricalRoom',  # 通风井 → 设备间（同类管井）
}

#: 走廊系列。GT 里靠拓扑区分主/次/公共走廊，但**轮廓提取根本区不出这三类**
#: （实测：GT 160/140/68，RGP 导出的训练集里只有 7/62/96）—— 三类在训练标签里
#: 就是噪声，留在标签空间里只会拖垮 macro-F1。``CORE`` 空间把它们合并。
CORRIDOR_LABELS: frozenset[str] = frozenset({
    'MainCorridor', 'SecondaryCorridor', 'PublicCorridor', 'Corridor',
})
CORRIDOR_TARGET = 'Corridor'


def folding_for(space: str = DEFAULT_SPACE,
                extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """按标签空间取折叠规则。

    ``FULL``
        只做最小折叠（:data:`LONG_TAIL_FOLDING`），尽量保留 GT 的原始类别，
        使结果能与 ``LLMMultiStage`` / ``TextMatching`` 直接对比。
    ``CORE``
        额外把走廊三类合成 ``Corridor``（见 :data:`CORRIDOR_LABELS`）。
    """
    rules = dict(LONG_TAIL_FOLDING)
    if space.upper() == 'CORE':
        for name in CORRIDOR_LABELS:
            rules[name] = CORRIDOR_TARGET
        rules.pop(CORRIDOR_TARGET, None)      # 目标自身不需要规则
    if extra:
        rules.update(extra)
    return rules


def fold_label(raw: str, folding: Mapping[str, str] | None = None) -> str:
    """把原始类型名折叠成规范名（幂等）。"""
    rules = LONG_TAIL_FOLDING if folding is None else folding
    return rules.get(raw, raw)


def resolve_classes(observed: Iterable[str], space: str = DEFAULT_SPACE,
                    folding: Mapping[str, str] | None = None,
                    min_samples: int = MIN_SAMPLES) -> tuple[str, ...]:
    """由观测到的类型集合解析出**排序后**的类别列表（保证跨次运行一致）。

    Args:
        observed: 折叠后的类型名可重复序列（用于统计频次）。
        space: ``FULL`` 或 ``CORE``。
        folding: 自定义折叠规则。
        min_samples: 低于该频次的类别处理方式 —— ``FULL`` 下并入 ``Other``
            （保留一个“其他”出口），``CORE`` 下直接排除（追求可学性）。
    """
    counts: dict[str, int] = {}
    for name in observed:
        counts[name] = counts.get(name, 0) + 1

    kept = {n for n, c in counts.items() if c >= min_samples}
    if space.upper() == 'CORE':
        return tuple(sorted(kept))
    if len(kept) < len(counts):
        kept.add('Other')
    return tuple(sorted(kept))


@dataclass
class LabelSpace:
    """类别索引 ↔ 类名的双向映射（随权重落盘）。"""

    classes: tuple[str, ...]
    space: str = DEFAULT_SPACE
    folding: dict[str, str] = field(default_factory=lambda: dict(LONG_TAIL_FOLDING))

    # ------------------------------------------------------------------ 构造
    @classmethod
    def fit(cls, raw_labels: Iterable[str], space: str = DEFAULT_SPACE,
            folding: Mapping[str, str] | None = None,
            min_samples: int = MIN_SAMPLES) -> 'LabelSpace':
        """从原始标签序列确定类别表。"""
        space = space.upper()
        rules = folding_for(space, folding)
        folded = [fold_label(lb, rules) for lb in raw_labels]
        return cls(classes=resolve_classes(folded, space, rules, min_samples),
                   space=space, folding=rules)

    def with_classes(self, classes: Iterable[str]) -> 'LabelSpace':
        """派生一个只保留指定类别的标签空间（顺序重排为字典序，保证确定性）。

        用于 ``CORE`` 的第二遍：先用 GT 标签定出候选类别表，再按导出数据里
        各类的**实际样本数**把样本过少的整类丢掉。
        """
        allowed = set(classes)
        return LabelSpace(classes=tuple(sorted(allowed)), space=self.space,
                          folding=dict(self.folding))

    # ------------------------------------------------------------------ 映射
    @property
    def n_class(self) -> int:
        return len(self.classes)

    @property
    def index_of(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.classes)}

    def encode(self, raw_label: str) -> int | None:
        """原始（未折叠）类型名 → 类别下标；不属于本标签空间时返回 ``None``。"""
        return self.index_of.get(fold_label(raw_label, self.folding))

    def decode(self, index: int) -> str:
        if 0 <= index < len(self.classes):
            return self.classes[index]
        return 'Other'

    def decode_many(self, indices: Sequence[int]) -> list[str]:
        return [self.decode(int(i)) for i in indices]

    # ------------------------------------------------------------------ 持久化
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'space': self.space, 'classes': list(self.classes),
                       'folding': self.folding}, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'LabelSpace':
        if not os.path.isfile(path):
            raise FileNotFoundError(
                '标签空间文件不存在: %s\n'
                '它必须与权重一同保存/加载（类别下标错位会静默给出错误结果）。' % path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return cls(classes=tuple(data['classes']), space=data.get('space', DEFAULT_SPACE),
                   folding=dict(data.get('folding') or LONG_TAIL_FOLDING))

    @classmethod
    def load_or_none(cls, path: str) -> 'LabelSpace | None':
        return cls.load(path) if path and os.path.isfile(path) else None
