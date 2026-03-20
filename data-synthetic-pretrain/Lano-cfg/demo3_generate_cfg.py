# 版权声明（c）Meta Platforms, Inc. 及其关联方。
#
# 作者：Zeyuan Allen-Zhu
#
#
# 在本 demo3 中，我们展示：
# 1) 如何加载 CFG 配置并可视化；
# 2) 如何从 CFG 生成一条序列（T 符号）；
# 3) 如何基于合法 CFG 序列计算真实 next-token 预测准确率。
#
# 备注：该流程对 probing 任务很有用，因为通常需要 NT 符号信息作为探测标签。
# 
"""
Demo3（生成 + 分布分析模板）

该示例与 Demo2 结构接近，重点在于给出一个可复用脚手架：
- 生成合法 CFG 序列；
- 计算真实 next-token 分布；
- 进一步做 KL 或其他分布级评估。
"""

from data_cfg import CFG_Config
import random
import numpy as np

if __name__ == '__main__':
    # =============================
    # 1) 加载并打印 CFG
    # =============================
    # 加载配置文件，例如 cfg3f
    config = CFG_Config.from_graph("configs/cfg3f.json")

    # 打印 CFG 规则，便于人工检查文法结构
    config.print_graph()

    # ===========================================
    # 2) 生成一条合法终结符序列（仅 T 符号）
    # ===========================================
    rng = random.Random(7711)  # 注意：这里使用 Python 的 random，而不是 numpy 的随机数
    seq = config.generate_onedata_pure(rng)
    print("This is a generated sequence (without EOS/BOS)", seq)
    # 示例输出（cfg3f）= [3, 3, 3, 3, 1, 1, 2, 1, 3, 1, 2, 3, 2, 2, 1, 2, 3, 3, 1, 2, 1, 3, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 3, 2, 2, 1, 3, 3, 1, 1, 2, 1, 1, 2, 1, 3, 2, 3, 3, 3, 1, 2, 1, 2, 1, 2, 3, 3, 1, 3, 3, 1, 3, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 1, 1, 3, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 3, 3, 2, 2, 1, 3, 1, 2, 3, 2, 2, 1, 2, 1, 3, 1, 2, 1, 1, 3, 2, 2, 1, 1, 3, 1, 2, 3, 3, 1, 2, 1, 3, 3, 1, 2, 1, 1, 3, 2, 3, 2, 2, 3, 1, 2, 3, 3, 3, 1, 2, 3, 1, 1, 3, 1, 1, 1, 1, 1, 2, 3, 1, 1, 3, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 3, 3, 3, 3, 1, 2, 3, 3, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 2, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1]
    # 示例输出（cfg3k）= [1, 3, 1, 1, 3, 1, 2, 1, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 2, 3, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 3, 3, 3, 1, 1, 1, 1, 3, 2, 3, 3, 2, 3, 2, 1, 3, 1, 2, 2, 2, 3, 2, 3, 2, 1, 3, 3, 1, 1, 2, 2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 2, 3, 1, 3, 1, 2, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 2, 2, 2, 3, 3, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 3, 2, 3, 3, 3, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 3, 1, 2, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 1, 3, 1, 3, 3, 3, 1, 1, 3, 1, 2, 1, 2, 3, 3, 3, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 3, 1, 3, 3, 2, 1, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 2, 3, 3, 3, 3, 2, 3, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1, 2, 2, 3, 2, 1, 1, 3, 1, 3, 3, 1, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 1, 2, 1, 2, 2, 3, 3, 2, 1, 3, 3, 1, 3, 1, 1, 2, 2, 1, 1, 2, 3, 3, 2, 3, 2, 1, 1, 3, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 3, 1, 2, 2, 1, 3, 1, 2, 2, 1, 1, 1, 3, 3, 2, 1, 2, 3, 3, 3, 1, 1, 2, 3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 3, 3, 2, 1, 3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 1, 2, 2, 3, 1, 2, 1, 3, 2, 3, 1, 3, 1, 2, 3, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 3, 2, 3, 2, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 1, 3, 3, 2, 3, 2, 3, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 3, 2, 1, 1, 3, 1, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2, 1, 1, 3, 1, 1, 3, 1, 3, 3, 3, 3, 1, 2, 2, 3, 3, 1, 2, 2, 3, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 3, 3, 2, 1, 1, 3, 1, 2, 1, 2, 3, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 1, 2, 3, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 3, 1, 3, 1, 3, 2, 3, 2, 3, 1, 3, 1, 2, 2, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3, 3, 2, 1, 2, 1, 3, 3, 2, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 3, 1, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 1]

    ############################################################################################################
    #  以下是更高级的用法：如果只想看基础流程，可以先跳过
    ############################################################################################################

    # KL 散度相关 DP（除准确率之外，用于评估分布层面的接近程度）
    if True:
        # 3) 基于前缀，使用高精度 DP 计算真实 next-token 条件分布
        # 警告：该函数要求 seq 是一条“合法 CFG 序列”
        # 提示：低精度版本 config.solve_dp_prob 速度更快，cfg3f 常可用；
        #      但对 cfg3k 等更长序列，建议使用 solve_dp_prob_highprecision
        target_dist, _ = config.solve_dp_prob_highprecision(seq, debug=False)  # 可改为 debug=True 查看 DP 细节日志
        target_dist = np.array(target_dist)
        np.set_printoptions(suppress=True)
        print()
        print(f"Next is the groud-truth next-token distribution conditioning on the prefix. If vocab size |T|=3 then each row has 4 numbers, first is EOS probability); each row sum up to 1.")
        print(target_dist[:20])

        # 计算 KL(target_dist, your_dist)：
        #   其中 your_dist 是你的语言模型输出的 next-token 预测分布（例如由 logits 归一化得到）
        #   这里演示时，先用 target_dist 拷贝一份并做轻微改动
        your_dist = target_dist.copy()
        your_dist[-1] = [1,0,0,0] # 假设模型在最后一个位置以 100% 概率预测 EOS
        # 4) 计算 KL(target || your)
        epsilon = 1e-5
        target_dist1 = (target_dist + epsilon) / (target_dist + epsilon).sum(axis=1, keepdims=True)
        your_dist1 = (your_dist + epsilon) / (your_dist + epsilon).sum(axis=1, keepdims=True)
        KV_div = np.mean(np.sum(target_dist * (np.log(target_dist1) - np.log(your_dist1)), axis=1))
        print(f"KL-divergence between target_dist and your_dist = {KV_div:.6f}")
        # 示例输出 = "KL-divergence between target_dist and your_dist = 0.015178"

    # 慢速 DP 的完整用法 --- config.solve_dp_noneq
    if False:
        count, dp_sol, counts, possibility = config.solve_dp_noneq(seq, no_debug=True)   # 可改 no_debug=False 查看详细 DP 过程
        print(count, counts, possibility)
        # 示例输出 = 0 [0, 0, 0, 0, 0, 0, 0] 1
        # count = 为满足 CFG 至少需要修改的 T 符号数（若无解则为 10000）
        # counts = 每一层级需要修改的符号数
        # possibility = 在该 CFG 下生成此序列的不同方式数（若为 1，表示解析唯一）
        # dp_sol = 满足 CFG 且与原序列“修改 token 数最少”的最优序列

        seq = [random.randint(1, config.num_sym) for _ in range(len(seq))]
        count, dp_sol, counts, possibility = config.solve_dp_noneq(seq, no_debug=True)   # 可改 no_debug=False 查看详细 DP 过程
        print(count, counts, possibility)
        # cfg3f 示例输出 = 50 [50, 48, 37, 20, 8, 3, 1] 18979019280
        # 含义：对同长度随机序列，至少要修改 50 个符号（不含插入/删除）才能满足 CFG
        # 其中：T 层修改 50 个，倒数第二层 NT 修改 48 个，倒数第三层 NT 修改 37 个，以此类推
        # 最后的 18979019280 是达到该最优修改数的方式数粗略估计（未去重）

    # 快速 DP 的完整用法 --- config.solve_dp_noneq_fast
    #   （慢速 DP 的弱化版：只能“验证是否满足 CFG”，不能求最小修改距离）
    if False:
        count, dp_sol, _, possibility = config.solve_dp_noneq_fast(seq, no_debug=True)   # 可改 no_debug=False 查看详细 DP 过程
        print(count, possibility)
        # 示例输出 = 0 1
        # count == 0 表示该序列满足 CFG；count == 10000 表示不满足 CFG
        # possibility = 1 表示该序列解析唯一（在给定 CFG 下只有一种生成方式）

        seq[0] = seq[0]%3+1
        count, dp_sol, _, possibility = config.solve_dp_noneq_fast(seq, no_debug=True)   # 可改 no_debug=False 查看详细 DP 过程
        print(count, _, possibility)
        # 示例输出 = 10000 None
        # count = 10000 表示该序列不满足 CFG。

