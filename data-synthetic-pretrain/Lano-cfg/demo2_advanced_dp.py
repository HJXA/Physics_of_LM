# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
#
# In this demo 2 we show:
# 1) how to load a CFG config and visualize it;
# 2) how to generate a sequence (T symbols) from the CFG;
# 3) how to compute the ground-truth next-token prediction accuracies based on a valid CFG sequence.
#    and how to compute the KL divergence between the ground-truth and a model's next-token prediction distribution (softmax logits).
# 4) how to compute (via SLOW DP) the minimum editing distance (i.e., token change, excluding insertion/deletion) from a valid CFG sequence
# 5) demonstrate that the FAST DP is simply a weakened version of SLOW DP, without computing minimum editing distance
#
# COMMENT: 3) is how we compute entropy / KL-divergence in the Physics of Language Model, Part 1 paper.
#          4) is not used in the paper, but can help estimate a model's correctness if it fails to generate fully-correct CFG sequence.
#          5) is identical to demo0, but is given here as a comparison to the SLOW DP.
# 
from data_cfg import CFG_Config
import random
import numpy as np

if __name__ == '__main__':
    # Load a config file, say cfg3f.
    config = CFG_Config.from_graph("configs/cfg3f.json")

    # Visualize the CFG rules (for fun)
    config.print_graph()

    # Generate a valid sequence
    rng = random.Random(7711)  # NOT numpy rng
    seq = config.generate_onedata_pure(rng)
    print("This is a generated sequence (without EOS/BOS)", seq)
    # output (from cfg3f) = [3, 3, 3, 3, 1, 1, 2, 1, 3, 1, 2, 3, 2, 2, 1, 2, 3, 3, 1, 2, 1, 3, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 3, 2, 2, 1, 3, 3, 1, 1, 2, 1, 1, 2, 1, 3, 2, 3, 3, 3, 1, 2, 1, 2, 1, 2, 3, 3, 1, 3, 3, 1, 3, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 1, 1, 3, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 3, 3, 2, 2, 1, 3, 1, 2, 3, 2, 2, 1, 2, 1, 3, 1, 2, 1, 1, 3, 2, 2, 1, 1, 3, 1, 2, 3, 3, 1, 2, 1, 3, 3, 1, 2, 1, 1, 3, 2, 3, 2, 2, 3, 1, 2, 3, 3, 3, 1, 2, 3, 1, 1, 3, 1, 1, 1, 1, 1, 2, 3, 1, 1, 3, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 3, 3, 3, 3, 1, 2, 3, 3, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 3, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 2, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1]
    # output (from cfg3k) = [1, 3, 1, 1, 3, 1, 2, 1, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 2, 3, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 3, 3, 3, 1, 1, 1, 1, 3, 2, 3, 3, 2, 3, 2, 1, 3, 1, 2, 2, 2, 3, 2, 3, 2, 1, 3, 3, 1, 1, 2, 2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 2, 3, 1, 3, 1, 2, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 2, 2, 2, 3, 3, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 3, 2, 3, 3, 3, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 3, 1, 2, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 1, 3, 1, 3, 3, 3, 1, 1, 3, 1, 2, 1, 2, 3, 3, 3, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 3, 1, 3, 3, 2, 1, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 2, 3, 3, 3, 3, 2, 3, 2, 1, 2, 1, 3, 3, 3, 3, 1, 1, 2, 2, 3, 2, 1, 1, 3, 1, 3, 3, 1, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 1, 2, 1, 2, 2, 3, 3, 2, 1, 3, 3, 1, 3, 1, 1, 2, 2, 1, 1, 2, 3, 3, 2, 3, 2, 1, 1, 3, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 3, 1, 2, 2, 1, 3, 1, 2, 2, 1, 1, 1, 3, 3, 2, 1, 2, 3, 3, 3, 1, 1, 2, 3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 3, 3, 2, 1, 3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 1, 2, 2, 3, 1, 2, 1, 3, 2, 3, 1, 3, 1, 2, 3, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 3, 2, 3, 2, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 1, 3, 3, 2, 3, 2, 3, 1, 3, 1, 3, 1, 2, 3, 2, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 3, 2, 1, 1, 3, 1, 2, 1, 3, 3, 2, 1, 2, 3, 2, 3, 2, 1, 1, 3, 1, 1, 3, 1, 3, 3, 3, 3, 1, 2, 2, 3, 3, 1, 2, 2, 3, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 1, 3, 3, 2, 1, 1, 3, 1, 2, 1, 2, 3, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 3, 1, 2, 3, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 3, 1, 3, 1, 3, 2, 3, 2, 3, 1, 3, 1, 2, 2, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 3, 3, 2, 1, 2, 1, 3, 3, 2, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 3, 1, 3, 1, 3, 1, 1, 3, 1, 1, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 1]

    ############################################################################################################
    #  Below are more advanced use cases, don't read unless you want to do more advance analysis
    ############################################################################################################

    # KL-divergence DP (for evaluating the distribution closeness, in addition to accuracy)
    if True:
        # Using DP to compute the ground-truth next-token distribution conditioning on the prefix.
        # WARNING: seq must be a valid sequence for this function call
        # TIP: I also have a low-precision (faster) function config.solve_dp_prob that will work for cfg3f, 
        #      but for longer sequences like cfg3k you may want to use config.solve_dp_prob_highprecision
        target_dist, _ = config.solve_dp_prob_highprecision(seq, debug=False)  # Can turn on debug=True for verbosity to see the DP progress
        target_dist = np.array(target_dist)
        np.set_printoptions(suppress=True)
        print()
        print(f"Next is the groud-truth next-token distribution conditioning on the prefix. If vocab size |T|=3 then each row has 4 numbers, first is EOS probability); each row sum up to 1.")
        print(target_dist[:20])

        # COMPUTE KL-div(target_dist, your_dist):
        #    where your_dist is the next-token prediction distribution (i.e., logits) that your language model outputs
        #    In this example, let us take your_dist = target_dist with some minor changes
        your_dist = target_dist.copy()
        your_dist[-1] = [1,0,0,0] # say that your code predicts EOS with 100% probability for the last token
        epsilon = 1e-5
        target_dist1 = (target_dist + epsilon) / (target_dist + epsilon).sum(axis=1, keepdims=True)
        your_dist1 = (your_dist + epsilon) / (your_dist + epsilon).sum(axis=1, keepdims=True)
        KV_div = np.mean(np.sum(target_dist * (np.log(target_dist1) - np.log(your_dist1)), axis=1))
        print(f"KL-divergence between target_dist and your_dist = {KV_div:.6f}")
        # output = "KL-divergence between target_dist and your_dist = 0.015178"

    # FULL USE of SLOW DP --- config.solve_dp_noneq
    if False:
        count, dp_sol, counts, possibility = config.solve_dp_noneq(seq, no_debug=True)   # Can turn on no_debug=False for verbosity to see the DP progress
        print(count, counts, possibility)
        # output = 0 [0, 0, 0, 0, 0, 0, 0] 1
        # count = the minimum number of T symbols need to be changed to satisfy the CFG (or count==10000 if no solution)
        # counts = the number of symbols that need to be changed per CFG level to satisfy the CFG
        # possibility = the number of different ways to generate this sequence from the given CFG (if it is 1, then the CFG has a unique parsing)
        # dp_sol = the sequence that satisfies the CFG and is ``closest'' to the given sequence in terms of number of token changes

        seq = [random.randint(1, config.num_sym) for _ in range(len(seq))]
        count, dp_sol, counts, possibility = config.solve_dp_noneq(seq, no_debug=True)   # Can turn on no_debug=False for verbosity to see the DP progress
        print(count, counts, possibility)
        # example output (for cfg3f) = 50 [50, 48, 37, 20, 8, 3, 1] 18979019280
        # This means, from a random sequence of the given length, you need to modify at least 50 symbols (without insert/delte) to make it satisfy the CFG
        # In this solution, you modify 50 T symbols, 48 NT symbols on the second to last layer, 37 NT symbols on the third to last layer, etc.
        # The last number 18979019280 is just a rough (since I'm not removing duplicates) estimate for how many possibilities to reach this optimal change 50. 

    # FULL USE of FAST DP --- config.solve_dp_noneq_fast
    #   (a weaker variant of SLOW DP, can only VERIFY but not to compute the minimum number of changes)
    if False:
        count, dp_sol, _, possibility = config.solve_dp_noneq_fast(seq, no_debug=True)   # Can turn on no_debug=False for verbosity to see the DP progress
        print(count, possibility)
        # output = 0 1
        # count == 0 means this sequence satisfies the CFG; 10000 means not satisfy CFG
        # possibility = 1 means this sequence has a unique parsing (i.e., it is the only way to generate this sequence from the given CFG)

        seq[0] = seq[0]%3+1
        count, dp_sol, _, possibility = config.solve_dp_noneq_fast(seq, no_debug=True)   # Can turn on no_debug=False for verbosity to see the DP progress
        print(count, _, possibility)
        # output = 10000 None
        # count = 1000 means this sequence does not satisfy the CFG.

