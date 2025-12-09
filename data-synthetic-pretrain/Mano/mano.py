# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
# This gives the necessary code to generate our [Mano] synthetic arithmetics datasets used in [Physics of language models: Part 4.1]
#


def encode_pure_arithmetic(rng, qids, lens, ops, knowledge_augment = True):
    this_idx = rng.randint(0, len(lens)-1)
    this_len = lens[this_idx]
    this_qid = qids[this_idx]

    def gen(ll, ops):
        nonlocal this_people_count, this_people
        if ll==0:
            vid = rng.randint(0, value_mod-1)
            return [5000+vid], [5000+vid], vid
        op = rng.choice(ops)
        l1 = rng.randint(0, ll-1)
        ans1 = gen(l1, ops)
        ans2 = gen(ll-1-l1, ops)

        opid = None
        if op == '+': opid = rng.choice([5, 1] if knowledge_augment else [1])
        if op == '-': opid = rng.choice([6, 2] if knowledge_augment else [2])
        if op == '*': opid = rng.choice([7, 3] if knowledge_augment else [3])
        if op == '/': opid = rng.choice([8, 4] if knowledge_augment else [4])
        # if op == '+': opid = 5
        # if op == '-': opid = 6
        # if op == '*': opid = 7
        # if op == '/': opid = 8

        seq1 = [opid] + ans1[0] + ans2[0]
        seq2 = [opid] + ans1[1] + ans2[1]
        if ans1[2] is None or ans2[2] is None:
            seq3 = None
        else:
            if op == '+': seq3 = (ans1[2] + ans2[2]) % value_mod
            if op == '-': seq3 = (ans1[2] - ans2[2] + value_mod) % value_mod
            if op == '*': seq3 = (ans1[2] * ans2[2]) % value_mod
            if op == '/':
                if ans2[2] == 0:
                    seq3 = None
                else:
                    seq3 = (ans1[2] * pow(ans2[2],-1,value_mod)) % value_mod
        return seq1, seq2, seq3

    while True:
        this_people = [rng.randint(0, value_mod-1) for _ in range(this_len+1)]
        this_people_count = 0
        question, _, ans = gen(this_len, ops)
        if ans is not None: break
    assert 0<=ans and ans<value_mod

    text = [bos_token_id - (rng.randint(0,1) if knowledge_augment else 0)]
    pid = 1

    text += [this_qid*10 + (rng.randint(0,1) if knowledge_augment else 0)]

    text += question

    text += [this_qid*10 + 4 + (rng.randint(0,1) if knowledge_augment else 0)]
    text += [5000+ans]

    return text
    
        
        
L = 10         # we used L=10,13,16 in the paper
ttype = 'asm'  # addition + subtraction + multiplication
value_mod = 23

knowledge_augment = True  # doesn't quite matter based on my experiments
bos_token_id = 9999


ccount = None
ops = []
if 'a' in ttype: ops+=['+']
if 's' in ttype: ops+=['-']
if 'm' in ttype: ops+=['*']
if 'd' in ttype: ops+=['/']
assert len(ops)>0

### During testing, we evaluate at exact len=L, and the code is not provided here
print(encode_pure_arithmetic(random.Random(42), qids=[a for a in range(1,L+1)], lens=[a for a in range(1,L+1)], ops=ops, knowledge_augment=knowledge_augment))
