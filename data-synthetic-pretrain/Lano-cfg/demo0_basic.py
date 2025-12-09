# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
#
# In this demo 0 we show:
# 1) how to load a CFG config, 
# 2) how to visualize the CFG rules,
# 3) how to generate a sequence from the CFG,
# 4) how to verify the generated sequence against the CFG (yes or no).
# This should be enough for most of the basic use cases.
# 
from data_cfg import CFG_Config
import random

if __name__ == '__main__':
    # Load a config file, say cfg3k.
    config = CFG_Config.from_graph("configs/cfg3k.json")

    # Visualize the CFG rules (for fun)
    config.print_graph()

    # !!! TL;DR - generation !!!
    rng = random.Random(7711)  # NOT numpy rng
    seq = config.generate_onedata_pure(rng)
    print("This is a generated sequence (without EOS/BOS)", seq)
    # output (from cfg3f) = [3, 3, 3, 3, 1, 1, 2, 1, 3, 1, 2, 3, 2, 2, 1, 2, 3, 3, 1, 2, 1, 3, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 3, 2, 2, 1, 3, 3, 1, 1, 2, 1, 1, 2, 1, 3, 2, 3, 3, 3, 1, 2, 1, 2, 1, 2, 3, 3, 1, 3, 3, 1, 3, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 1, 1, 3, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 3, 3, 2, 2, 1, 3, 1, 2, 3, 2, 2, 1, 2, 1, 3, 1, 2, 1, 1, 3, 2, 2, 1, 1, 3, 1, 2, 3, 3, 1, 2, 1, 3, 3, 1, 2, 1, 1, 3, 2, 3, 2, 2, 3, 1, 2, 3, 3, 3, 1, 2, 3, 1, 1, 3, 1, 1, 1, 1, 1, 2, 3, 1, 1, 3, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 3, 3, 3, 3, 1, 2, 3, 3, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 2, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1]
    # output (from cfg3k) = [1, 3, 1, 1, 3, 1, 2, 1, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 2, 3, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 3, 3, 3, 1, 1, 1, 1, 3, 2, 3, 3, 2, 3, 2, 1, 3, 1, 2, 2, 2, 3, 2, 3, 2, 1, 3, 3, 1, 1, 2, 2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 2, 3, 1, 3, 1, 2, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 2, 2, 2, 3, 3, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 3, 2, 3, 3, 3, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 3, 1, 2, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 1, 3, 1, 3, 3, 3, 1, 1, 3, 1, 2, 1, 2, 3, 3, 3, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 3, 1, 3, 3, 2, 1, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 2, 3, 3, 3, 3, 2, 3, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1, 2, 2, 3, 2, 1, 1, 3, 1, 3, 3, 1, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 1, 2, 1, 2, 2, 3, 3, 2, 1, 3, 3, 1, 3, 1, 1, 2, 2, 1, 1, 2, 3, 3, 2, 3, 2, 1, 1, 3, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 3, 1, 2, 2, 1, 3, 1, 2, 2, 1, 1, 1, 3, 3, 2, 1, 2, 3, 3, 3, 1, 1, 2, 3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 3, 3, 2, 1, 3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 1, 2, 2, 3, 1, 2, 1, 3, 2, 3, 1, 3, 1, 2, 3, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 3, 2, 3, 2, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 1, 3, 3, 2, 3, 2, 3, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 3, 2, 1, 1, 3, 1, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2, 1, 1, 3, 1, 1, 3, 1, 3, 3, 3, 3, 1, 2, 2, 3, 3, 1, 2, 2, 3, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 3, 3, 2, 1, 1, 3, 1, 2, 1, 2, 3, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 1, 2, 3, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 3, 1, 3, 1, 3, 2, 3, 2, 3, 1, 3, 1, 2, 2, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3, 3, 2, 1, 2, 1, 3, 3, 2, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 3, 1, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 1]

    # !!! TL;DR - verification (DP could take a few seconds for cfg3f, or <=30 seconds for cfg3k) !!!
    correct, _, _, _ = config.solve_dp_noneq_fast(seq, no_debug=True) 
    # correct = 0 or 10000, if 0 means the sequence satisfies the CFG, if 10000 means it does not
    assert correct==0, f"Generated sequence {seq} must satisfy the CFG."

    # !!! TL;DR - verification2 (DP could take a few seconds for cfg3f, or <=30 seconds for cfg3k) !!!
    seq[0] = seq[0]%3+1
    correct, _, _, _ = config.solve_dp_noneq_fast(seq, no_debug=True) 
    print(correct)   # most likely output = 10000, meaning does not satisfy the CFG after flipping the first token

