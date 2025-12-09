# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
#
# In this demo 1 we show:
# 1) how to load a CFG config and visualize it;
# 2) how to generate a sequence (T symbols) from the CFG along with its correct parsing tree (NT symbols);
# 3) how to verify the generated sequence against the CFG (yes or no), and obtain a valid parsing tree (if yes).
#
# COMMENT: This can be useful for users interested in probing 
#      --- because you need to use the NT symbol (parsing tree) as labels for linear probing
# 
from data_cfg import CFG_Config
import random

if __name__ == '__main__':
    # Load a config file, say cfg3k.
    config = CFG_Config.from_graph("configs/cfg3k.json")

    # Visualize the CFG rules (for fun)
    config.print_graph()

    rng = random.Random(7711)  # NOT numpy rng
    seq = config.generate_onedata_pure(rng)
    rng = random.Random(7711)
    seq_tree = config.generate_onedata(rng)
    assert seq == [seq_tree[i][0] for i in range(len(seq_tree))], "seq_tree[i][0] is exactly the T symbols"
    print("This is a generated sequence (without EOS/BOS)", seq)
    # # output (from cfg3f) = [3, 3, 3, 3, 1, 1, 2, 1, 3, 1, 2, 3, 2, 2, 1, 2, 3, 3, 1, 2, 1, 3, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 3, 2, 2, 1, 3, 3, 1, 1, 2, 1, 1, 2, 1, 3, 2, 3, 3, 3, 1, 2, 1, 2, 1, 2, 3, 3, 1, 3, 3, 1, 3, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 1, 1, 3, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 3, 3, 2, 2, 1, 3, 1, 2, 3, 2, 2, 1, 2, 1, 3, 1, 2, 1, 1, 3, 2, 2, 1, 1, 3, 1, 2, 3, 3, 1, 2, 1, 3, 3, 1, 2, 1, 1, 3, 2, 3, 2, 2, 3, 1, 2, 3, 3, 3, 1, 2, 3, 1, 1, 3, 1, 1, 1, 1, 1, 2, 3, 1, 1, 3, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 3, 3, 3, 3, 1, 2, 3, 3, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 2, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1]
    # # output (from cfg3k) = [1, 3, 1, 1, 3, 1, 2, 1, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 2, 3, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 3, 3, 3, 1, 1, 1, 1, 3, 2, 3, 3, 2, 3, 2, 1, 3, 1, 2, 2, 2, 3, 2, 3, 2, 1, 3, 3, 1, 1, 2, 2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 2, 3, 1, 3, 1, 2, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 2, 2, 2, 3, 3, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 3, 2, 3, 3, 3, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 3, 1, 2, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 1, 3, 1, 3, 3, 3, 1, 1, 3, 1, 2, 1, 2, 3, 3, 3, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 3, 1, 3, 3, 2, 1, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 2, 3, 3, 3, 3, 2, 3, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1, 2, 2, 3, 2, 1, 1, 3, 1, 3, 3, 1, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 1, 2, 1, 2, 2, 3, 3, 2, 1, 3, 3, 1, 3, 1, 1, 2, 2, 1, 1, 2, 3, 3, 2, 3, 2, 1, 1, 3, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 3, 1, 2, 2, 1, 3, 1, 2, 2, 1, 1, 1, 3, 3, 2, 1, 2, 3, 3, 3, 1, 1, 2, 3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 3, 3, 2, 1, 3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 1, 2, 2, 3, 1, 2, 1, 3, 2, 3, 1, 3, 1, 2, 3, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 3, 2, 3, 2, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 1, 3, 3, 2, 3, 2, 3, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 3, 2, 1, 1, 3, 1, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2, 1, 1, 3, 1, 1, 3, 1, 3, 3, 3, 3, 1, 2, 2, 3, 3, 1, 2, 2, 3, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 3, 3, 2, 1, 1, 3, 1, 2, 1, 2, 3, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 1, 2, 3, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 3, 1, 3, 1, 3, 2, 3, 2, 3, 1, 3, 1, 2, 2, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3, 3, 2, 1, 2, 1, 3, 3, 2, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 3, 1, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 1]

    print("This is a generated sequence with parsing tree (T and NT symbols)", seq_tree)
    for i in range(20):
        print(seq_tree[i])

    # OUTPUT (for cfg3k)= 
    # [1, -2, 7, -2, 11, -2, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [3, -3, 7, -2, 11, -2, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -4, 7, -2, 11, -2, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -5, 7, -3, 11, -2, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [3, -6, 7, -3, 11, -2, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -7, 7, -3, 11, -2, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [2, -8, 7, -4, 11, -2, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -9, 7, -4, 11, -2, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [3, -10, 8, -5, 12, -3, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -11, 8, -5, 12, -3, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -12, 8, -6, 12, -3, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -13, 8, -6, 12, -3, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [2, -14, 8, -6, 12, -3, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [2, -15, 7, -7, 12, -3, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -16, 7, -7, 12, -3, 15, -2, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -17, 8, -8, 12, -4, 13, -3, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -18, 8, -8, 12, -4, 13, -3, 18, -2, 20, -2, 24, -2, 25, -2]
    # [2, -19, 8, -8, 12, -4, 13, -3, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -20, 9, -9, 12, -4, 13, -3, 18, -2, 20, -2, 24, -2, 25, -2]
    # [1, -21, 9, -9, 12, -4, 13, -3, 18, -2, 20, -2, 24, -2, 25, -2]

    # The (positive) numbers at odd positions represent the T/NT symbols at different levels (from leave to root)
    #  E.g., the 1-st  token has T=1 and NT=(7, 11, 15, 18, 20, 24, 25)
    #  E.g., the 11-th token has T=1 and NT=(8, 12, 15, 18, 20, 24, 25)
    # IMPORTANT: those NT symbols are NOT sufficient in recoverring the parsing tree,
    #            because for instance when 12 -> 8 8 7, one needs to differentiate the two NT symbol 8's.

    # The (negative) numbers at even positions represent the indices of the T/NT symbols in the sequence.
    # Formally, if seq_tree[i][2*j+1]==-1-k (for k>=1) then it means the T/NT symbol seq_tree[i][2*j] is the k-th T/NT symbol at level j
    #  E.g., seq_tree[i][1]==-2-i because the T symbol at position i is always the i-th T symbol at level 0.
    #  E.g., seq_tree[i][3]==(-2,-2,-2,-3,-3,-3,-4,-4,-5,-5,....) means that the NT symbols at level 1 are (NT=7, NT=7, NT=7, NT=8) but each spans 3, 3, 2, 2 tokens respectively.
    #
    # Lemma (trivial): from seq_tree one can recover the parsing tree by simply grouping the T/NT symbols according to their levels.

    # Part 1 of our paper shows that:
    #  1) the NT symbols (odd ppositions) are stored in the last transformer layer --- up to linear transformation (linear probing)
    #  2) the NT boundary information (i.e., those i with seq_tree[i][2*j+1] != seq_tree[i+1][2*j+1]) are also stored in the last transformer layer

    correct, dp_sol, _, _ = config.solve_dp_noneq_fast(seq, no_debug=True) 
    # As we saw in demo0, correct = 0 or 10000 indicating if seq satisfies the CFG (0 for yes, 10000 for no)
    # Additionally, dp_sol is a valid parsing tree (if correct == 0) or None (if correct == 10000).
    assert correct==0, f"Generated sequence {seq} must satisfy the CFG."
    print("This is a valid parsing tree (NT symbols) for the generated sequence:")
    for i in range(20):
        print(dp_sol[i])
    # It could be that dp_sol == seq_tree, but if the CFG is ambiguous, then dp_sol could be different from seq_tree.

