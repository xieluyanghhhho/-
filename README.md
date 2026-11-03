# -
原创作者谢璐阳 豆包 deepseek ：个人独立思想实验架构   当前状态：思想实验完整定稿 · 可开源 · 可二次开发 · 可工程落地   核心宣言：高维空间换取时间，打通AGI与图灵测试的长记忆边界墙（不拆墙、只破壁）但是这个只是思想实验，下载后可以随便改代码和注释，豆包生成论文，deepseek思考93秒后写的代码，豆包大模型封装代码 2A智能体大作，
警告！使用文档插件记得提前原始的备份点 ，如果不备份数据库炸了后果自负！
本人完全不会写代码 只好这样子提供内容
https://www.doubao.com/thread/xsisJVZTcdH5FM6Bw
论文题目https://feishu.doubao.com/docx/DIeidU4bWo6lQpxURptcJSYX6nh
免责声明 从第八行开始 我代码完全看不懂 我绝对不是师承余胜军说Python！ 绝对不是师承余胜军说Python！
完全是拍砖引玉（有bug很正常因为我啥都不会，改代码随便改，要商用改动记得备份原始数据库！！！ 改错了了概不负责）
## 📄完整 hd_stalm.py（补全你粘贴的代码）
```python
"""
HD-STALM: 高维时空锚点长记忆系统 — 参考实现 (Reference Prototype)
=================================================================

对应《高维时空锚点长记忆系统》思想实验文档的工程原型。

流形维度分配 (总计 15890 维):
    [0     : 4089 ]  实体A 语义轴          (例: 猫)
    [4089  : 9169 ]  实体B 语义轴          (例: 狗)
    [9169  : 15059]  全局唯一时序基准轴
    [15059 : 15890]  高维拓扑骨架 / 隔离缓冲 / 容错空间

核心机制:
    1. 高维空间换时间        —— 时序有序插入高维流形
    2. 准零维收敛子锚阵列    —— 半稳定微小邻域, 不做全域奇点坍缩
    3. 人脑式双向记忆        —— 高权重锁定 + 低权重时序衰减
    4. 拓扑故障隔离          —— 锚点间隔离缓冲区, 单点故障不扩散
    5. 动态分片 + 预缓冲拆分 + 冷记忆归档
    6. AGI 上下文耗尽语义复活 —— 窗口清空后按时间线重建思维链

依赖: numpy
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "HDSTALMConfig",
    "ManifoldEmbedder",
    "MemoryItem",
    "SubAnchor",
    "Shard",
    "HDSTALM",
]


# =====================================================================
# 0. 配置
# =====================================================================
@dataclass
class HDSTALMConfig:
    """流形维度与行为参数"""

    # ---- 维度骨架 (严格 15890) ----
    dim_entity_a: int = 4089
    dim_entity_b: int = 5080
    dim_time: int = 5890
    dim_skeleton: int = 831

    # ---- 准零维子锚 ----
    neighborhood_radius: float = 0.12   # 子锚吸收半径 (极小于流形尺度 → 准零维)
    isolation_buffer: float = 0.03      # 拓扑隔离缓冲区, 必须 < neighborhood_radius
    max_items_per_anchor: int = 8       # 子锚容量, 超出触发预缓冲拆分
    max_anchors_per_shard: int = 12     # 分片容量, 超出触发动态分片

    # ---- 类人遗忘 ----
    hot_weight_threshold: float = 0.80  # 高于此权重 → 热记忆锁定, 不衰减
    cold_weight_threshold: float = 0.15 # 低于此权重 → 冷记忆归档
    decay_k: float = 0.01               # 衰减系数 (低权重衰减更快)

    # ---- 检索 ----
    top_k_anchors: int = 8              # 先按锚点粗筛, 再在锚内精排

    # ---------------- 派生属性 ----------------
    @property
    def total_dim(self) -> int:
        return (self.dim_entity_a + self.dim_entity_b
                + self.dim_time + self.dim_skeleton)

    @property
    def slice_a(self) -> slice:
        return slice(0, self.dim_entity_a)

    @property
    def slice_b(self) -> slice:
        s = self.dim_entity_a
        return slice(s, s + self.dim_entity_b)

    @property
    def slice_t(self) -> slice:
        s = self.dim_entity_a + self.dim_entity_b
        return slice(s, s + self.dim_time)

    @property
    def slice_s(self) -> slice:
        s = self.dim_entity_a + self.dim_entity_b + self.dim_time
        return slice(s, s + self.dim_skeleton)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# =====================================================================
# 1. 流形嵌入层: (实体A, 实体B, 时间) → 15890 维点位
# =====================================================================
class ManifoldEmbedder:
    """把一条记忆映射为 15890 维流形上的一个单位球面点位"""

    def __init__(self, cfg: HDSTALMConfig):
        self.cfg = cfg
        half = cfg.dim_time // 2
        # 时序基准轴: 多频傅里叶特征, 保证时间邻近 → 向量邻近
        self._freq = np.logspace(-3.0, -0.7, half)
        self._skeleton_gain = 0.1   # 骨架维降权, 只做拓扑扰动, 不主导距离

    # ---------- 实体语义轴 ----------
    @staticmethod
    def _ngrams(text: str, n: int = 2) -> List[str]:
        text = (text or "").strip()
        if not text:
            return ["<empty>"]
        if len(text) <= n:
            return [text]
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def _hash_embed(self, text: str, dim: int, seed: int) -> np.ndarray:
        """字符 2-gram + 整串 的确定性哈希嵌入, 共享子串 → 共享维度"""
        v = np.zeros(dim, dtype=np.float64)
        grams = self._ngrams(text) + [text or "<empty>"]
        for g in grams:
            h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if (h[4] & 1) else -1.0
            v[idx] += sign * (1.0 if len(g) > 1 else 0.5)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def entity_a(self, text: str) -> np.ndarray:
        return self._hash_embed(text, self.cfg.dim_entity_a, seed=1001)

    def entity_b(self, text: str) -> np.ndarray:
        return self._hash_embed(text, self.cfg.dim_entity_b, seed=2002)

    # ---------- 时序基准轴 ----------
    def time_axis(self, t: float) -> np.ndarray:
        x = float(t) * self._freq
        v = np.concatenate([np.sin(x), np.cos(x)])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # ---------- 拓扑骨架 / 隔离缓冲 ----------
    def skeleton(self, text: str, t: float) -> np.ndarray:
        h = hashlib.blake2b(f"{text}|{t}".encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(h, "little"))
        v = rng.standard_normal(self.cfg.dim_skeleton)
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v) * self._skeleton_gain

    # ---------- 总装 ----------
    def embed(self, a_text: str, b_text: str, t: float) -> np.ndarray:
        v = np.concatenate([
            self.entity_a(a_text),
            self.entity_b(b_text),
            self.time_axis(t),
            self.skeleton(f"{a_text}|{b_text}", t),
        ])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


# =====================================================================
# 2. 记忆单元
# =====================================================================
@dataclass
class MemoryItem:
    item_id: str
    timestamp: float
    content: str
    vector: np.ndarray          # 15890 维单位向量
    importance: float
    weight: float
    last_decay_ts: float
    locked: bool = False
    archived: bool = False
    access_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# 3. 准零维收敛子锚
# =====================================================================
@dataclass
class SubAnchor:
    anchor_id: str
    center: np.ndarray
    radius: float
    weight: float = 0.0
    items: List[MemoryItem] = field(default_factory=list)
    t_min: float = math.inf
    t_max: float = -math.inf
    corrupted: bool = False       # 故障隔离标记

    def add_item(self, item: MemoryItem) -> None:
        n = len(self.items)
        self.items.append(item)
        # 中心点: 单位球面上的滑动平均
        merged = (self.center * n + item.vector) / (n + 1)
        norm = float(np.linalg.norm(merged))
        self.center = merged / norm if norm > 1e-12 else item.vector
        # 收敛半径: 记录成员到中心的最大偏移
        d = float(np.linalg.norm(item.vector - self.center))
        self.radius = max(self.radius, d)
        self.t_min = min(self.t_min, item.timestamp)
        self.t_max = max(self.t_max, item.timestamp)
        self.weight += item.weight

    def mean_time(self) -> float:
        return 0.5 * (self.t_min + self.t_max) if self.items else 0.0


# =====================================================================
# 4. 分片 Shard（继续在此基础上补全剩余类与HDSTALM主类）
# ================
