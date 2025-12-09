# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
# This gives the necessary code to generate our [Mano] synthetic arithmetics datasets used in [Physics of language models: Part 4.1]
#


def generate_multi_token_words(rng, n: int,
    mini_vocab: int = 3,
    min_tlen: int = 5,
    max_tlen: int = 7
):
    def my_sample(length):
        toks = [rng.randint(1, mini_vocab) for _ in range(length)]
        toks[-1] += mini_vocab  # end of word
        return tuple(toks)      # tuples are hashable 

    words = set()
    while len(words) < n:
        length = rng.randint(min_tlen, max_tlen)
        word = my_sample(length)
        words.add(word)

    return [list(word) for word in words]    


rng = random.Random(42)

##
## QA=True means it's test data, in which we force n = N and k = powers of 2 (feel free to change this)
##
def depo_data(N, K, M=10, QA=False, separator = False, mini_vocab: int = 3, min_tlen: int = 5, max_tlen: int = 7):
    if QA:
        n = N
    else:
        _distribution_list_to_choose = list(range(3, N+1))
        power, bias = 1, pow(N, 0.5)
        p = [1.0 / (pow(i, power) + bias + 1e-12) for i in _distribution_list_to_choose]
        _distribution_p = [1.0 * x / sum(p) for x in p]
        n = rng.choices(_distribution_list_to_choose, weights=_distribution_p)[0]


    vals = generate_multi_token_words(rng, n, mini_vocab=mini_vocab, min_tlen=min_tlen, max_tlen=max_tlen)
    rng.shuffle(vals)
    order = rng.sample(range(n), n)

    text = [bos_token_id] # eos
    for i in range(n):
        if separator:
            text += [9700]
        v1 = vals[order[i]]
        v2 = vals[(order[i]+1)%n]
        text += v1 + v2
    list_label = [0] * len(text)

    m = M

    if QA:
        powers = [2**i for i in range(K.bit_length()) if 2**i <= K]
        if K==32:
            powers += [24]
        if powers[-1] != K:
            powers.append(K)
        
    first = True
    for idx in rng.sample(range(n), k=min(n, m)):
        if QA:
            k = rng.choice(powers)                        
        else:
            k = rng.randint(1,K)
        v1,v2 = vals[idx], vals[(idx + k) % n]
        text += [9000 + k] + v1 + [9500] + v2
        list_label += [0] * (len(v1)+1) + [1] * (len(v2)+1)
        first=False

    assert len(token_type)==len(list_label)

    ## list_labels = 0 or 1, where 1's are for answer tokens. We pretrain only on those answer tokens --- pretraining with all the tokens yield similar results, but is a factor slower.
    return {0:text, 'label':list_label}        





### This is Depo1
print(depo_data(N=375, K=8, mini_vocab=50, min_tlen=1, max_tlen=2))
print(depo_data(N=300, K=8, mini_vocab=50, min_tlen=1, max_tlen=2))
print(depo_data(N=225, K=8, mini_vocab=50, min_tlen=1, max_tlen=2))

### This is Depo2
print(depo_data(N=125,K=16))
print(depo_data(N=100,K=16))
print(depo_data(N=75,K=16))
