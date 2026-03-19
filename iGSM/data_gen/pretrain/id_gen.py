# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math_gen.problem_gen import Problem
from data_gen.prototype.id_gen import IdGen_PT

"""
预训练样本生成入口（标准版本）。

该文件将 `IdGen_PT` 以轻量参数分布封装为 `IdGen`，
用于生成 iGSM 的基础预训练样本。
"""

class IdGen(IdGen_PT):
    def __init__(self, max_op=10, max_edge=15, op=None, perm_level: str = None, detail_level: str = None, be_shortest: bool=True) -> None:
        """初始化预训练生成器（标准版）。"""
        super().__init__('light', 'light', max_op, max_edge, op, perm_level, detail_level, be_shortest)
    
    def gen_prob(self, ava_hash, p_format: str, problem: Problem=None):
        """生成一道题并编码为 token（继承父类完整实现）。"""
        super().gen_prob(ava_hash, p_format, problem=problem)

