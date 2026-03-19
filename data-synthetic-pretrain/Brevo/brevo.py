# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
# This gives the necessary code to generate our [Mano] synthetic arithmetics datasets used in [Physics of language models: Part 4.1]
#
"""
Brevo 数据生成脚本（拓扑排序任务）。

核心思路：
1) 构造一个带父依赖关系的 DAG；
2) 随机选取查询节点，只保留其可达子图；
3) 输出该子图的一个合法拓扑序作为答案；
4) 可选地将“节点”编码成多 token 伪词（multi=True）以提高任务复杂度。

该文件同时提供了输出解析函数，用于验证模型生成答案是否满足拓扑约束。
"""


def generate_multi_token_words(rng, n: int,
    mini_vocab: int = 3,
    min_tlen: int = 5,
    max_tlen: int = 7
):
    """生成 n 个不重复的“多 token 伪词”。

    每个伪词末尾 token 落在 (mini_vocab, 2*mini_vocab] 区间，
    用于作为“单词结束符”做切分。
    """
    def my_sample(length):
        """按指定长度采样一个可哈希伪词（tuple）。"""
        toks = [rng.randint(1, mini_vocab) for _ in range(length)]
        toks[-1] += mini_vocab  # end of word
        return tuple(toks)      # tuples are hashable 

    words = set()
    while len(words) < n:
        length = rng.randint(min_tlen, max_tlen)
        word = my_sample(length)
        words.add(word)

    return [list(word) for word in words]    


class TopoSortDepthStats:
    """拓扑排序任务生成与验证器。"""

    def __init__(self, n, vocab_size=125, max_in=4):
        """初始化图规模和约束。

        - n: 图节点数；
        - vocab_size: 节点 token 池大小；
        - max_in: 预留参数（当前主要由 degree_constraint 控制）。
        """
        self.n = n
        self.vocab_size = vocab_size
        self.max_in = max_in
        self.degree_constraint = 'A_dep_B_for_at_most_4_Bs_and_4_As_with_leaves_on_left'
        

    def generate_dag(self, rng):
        """生成带约束的 DAG。

        约束目标：每个节点最多依赖 4 个父节点，且每个父节点最多被 4 个子节点依赖。
        """
        nodes = rng.sample(range(1, self.vocab_size + 1), self.n)
        dag = defaultdict(list)
        if self.degree_constraint == 'A_dep_B_for_at_most_4_Bs_and_4_As_with_leaves_on_left':
            out_degree = defaultdict(int)
            leaves = rng.randint(1, (len(nodes)-1)//4+1)
            for i in range(leaves, len(nodes)):
                tgt = nodes[i]
                possible_parents = [src for src in nodes[:i] if out_degree[src] < 4]
                if not possible_parents:
                    continue  # no available parents with capacity
                num_parents = rng.randint(1, min(len(possible_parents), 4))
                #if i!=len(nodes)-1 and rng.randint(0,7)==0:
                #    num_parents = 0
                parents = rng.sample(possible_parents, num_parents)
                for parent in parents:
                    dag[tgt].append(parent)
                    out_degree[parent] += 1
        else:
            assert False, f"I removed other versions; different degree distributions may make the task too easy or degenerate"

        return nodes, dag

    def subtree_from_query(self, dag, query):
        """从 query 逆向搜可达父节点，提取查询相关子图。"""
        visited = set()
        stack = [query]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for parent in dag.get(node, []):
                    if parent not in visited:
                        stack.append(parent)
        filtered = defaultdict(list)
        for node in visited:
            for parent in dag.get(node, []):
                if parent in visited:
                    filtered[node].append(parent)
        for node in visited:
            _ = filtered[node]
        return filtered

    def topological_sort(self, dag, rng):
        """对给定子图执行随机化拓扑排序。"""
        indegree = {node: 0 for node in dag}
        for node in dag:
            for parent in dag[node]:
                indegree[parent] += 1
        queue = [node for node in dag if indegree[node] == 0]
        order = []
        while queue:
            node = queue.pop(rng.randint(0, len(queue) - 1))
            order.append(node)
            for parent in dag[node]:
                indegree[parent] -= 1
                if indegree[parent] == 0:
                    queue.append(parent)
        order.reverse()
        return order

    def compute_graph_depth(self, dag, query):
        """计算 query 到任一叶子节点的最短逆向距离。"""
        distance = {query: 0}
        queue = deque([query])
        while queue:
            node = queue.popleft()
            for parent in dag[node]:
                if parent not in distance:
                    distance[parent] = distance[node] + 1
                    queue.append(parent)
        leaves = [node for node in dag if len(dag[node]) == 0]
        if not leaves:
            return 0
        return min(distance.get(leaf, float('inf')) for leaf in leaves if leaf in distance)

    def generate_sample(self, rng):
        """生成一条样本：原图、答案拓扑序、以及查询深度。"""
        nodes, dag = self.generate_dag(rng)

        start_index = max(len(nodes) * 3 // 4, len(nodes) - 1)  # safe even if len(nodes) == 2
        candidate_nodes = nodes[start_index:]
        nonzero_degree_nodes = [node for node in candidate_nodes if len(dag[node]) > 0]
        query = rng.choice(nonzero_degree_nodes)

        subdag = self.subtree_from_query(dag, query)
        topo = self.topological_sort(subdag, rng)
        depth = self.compute_graph_depth(subdag, query)
        return dag, topo, depth


    def generate_tokens(self, rng, multi=False):
        """将图任务编码成 token 序列。

        返回值：
        - tokens: 输入 + 目标序列拼接后的 token；
        - token_type: 辅助标签（主要用于分析/可视化）；
        - list_label: 训练掩码，1 表示答案区间；
        - depth: 该样本查询深度。
        """
        dag, topo, depth = self.generate_sample(rng)
        query = topo[-1]

        if multi:
            all_node_ids = sorted(set(dag.keys()) | {p for ps in dag.values() for p in ps})
            id_to_index = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
            index_to_id = {idx: node_id for node_id, idx in id_to_index.items()}

            word_list = generate_multi_token_words(rng,
                n=len(all_node_ids),
                mini_vocab=4,
                min_tlen=2,
                max_tlen=4,
            )

            word_map = {node_id: word_list[id_to_index[node_id]] for node_id in all_node_ids}
    
            
        edges = [(p, c) for c, ps in dag.items() for p in ps]
        rng.shuffle(edges)


        tokens = [bos_token_id]
        for p, c in edges:
            if multi:
                tokens += word_map[p] + word_map[c]
            else:
                tokens += [p, c]
        if multi:
            tokens += [bos_token_id-2] + word_map[query] + [bos_token_id-1]
        else:
            tokens += [bos_token_id-2, query, bos_token_id-1]
        list_label = [0] * (len(tokens)-1)
        token_type = [-1, -2-depth] + [0] * (len(tokens)-2)
        fake_depth = depth if depth<=9 else 9
        if multi:
            first = True
            for node in topo:
                tokens += word_map[node]
                if first: 
                    token_type += [fake_depth+1] * len(word_map[node])
                    first = False
                else:
                    token_type += [0] * len(word_map[node])
        else:
            tokens += topo
            token_type += [fake_depth+1] + [0] * (len(topo)-1)  # only check if first token is correct, for training illustration only
        tokens += [eos_token_id]
        token_type += [0]
        list_label += [1] * (len(tokens) - len(list_label))
        assert len(tokens) == len(list_label) and len(list_label) == len(token_type), f"{len(tokens)} {len(list_label)} {len(token_type)}"
        return tokens, token_type, list_label, depth
       
    @staticmethod
    def parse_tokens(tokens):
        """解析并校验单 token 版本输出是否拓扑合法。"""
        if tokens[0] != bos_token_id or tokens[-1] != bos_token_id:
            return False, None, None
        try:
            idx_query = tokens.index(bos_token_id-2)
            idx_answer = tokens.index(bos_token_id-1)
        except ValueError:
            return False, None, None

        edge_tokens = tokens[1:idx_query]
        if len(edge_tokens) % 2 != 0:
            return False, None, None

        edges = [(edge_tokens[i], edge_tokens[i + 1]) for i in range(0, len(edge_tokens), 2)]
        query = tokens[idx_query + 1]
        topo = tokens[idx_answer + 1 : -1]

        # Build DAG: child ← [parents]
        dag = defaultdict(list)
        for parent, child in edges:
            dag[child].append(parent)

        # Step 1: compute reachable nodes (reverse DFS from query)
        reachable = set()
        stack = [query]
        while stack:
            node = stack.pop()
            reachable.add(node)
            for parent in dag.get(node, []):
                if parent not in reachable:
                    stack.append(parent)

        # Step 2: validate all nodes in topo are reachable from query
        if set(topo) != reachable:
            return False, query, topo

        # Step 3: validate topological order
        seen = set()
        for node in topo:
            for parent in dag.get(node, []):
                if parent not in seen:
                    return False, query, topo
            seen.add(node)

        # print("Correctness explanation:")
        # print(dag)
        # print("query=",query)
        # print(topo)
        # print("End of explanation")


        return True, query, topo


    # Can use the following trivial code to evaluate a model's output --- topsort answer is not unique
    @staticmethod
    def parse_tokens_multi(tokens, mini_vocab=4):
        """解析并校验 multi-token 版本输出是否拓扑合法。"""
        tokens = [a for a in tokens if a!=9700]
        if tokens[0] != bos_token_id or tokens[-1] != bos_token_id:
            return False, None, None
        try:
            idx_query = tokens.index(bos_token_id-2)
            idx_answer = tokens.index(bos_token_id-1)
        except ValueError:
            return False, None, None

        def split_words(token_seq):
            words = []
            word = []
            for tok in token_seq:
                word.append(tok)
                if mini_vocab < tok <= 2 * mini_vocab:
                    words.append(tuple(word))
                    word = []
            return words

        # 1. Parse all sections
        edge_tokens = tokens[1:idx_query]
        query_tokens = tokens[idx_query + 1 : idx_answer]
        answer_tokens = tokens[idx_answer + 1 : -1]

        edge_words = split_words(edge_tokens)
        query_word = tuple(split_words(query_tokens)[0])
        answer_words = split_words(answer_tokens)

        if len(edge_words) % 2 != 0:
            return False, None, None

        # 2. Assign unique IDs to each word
        all_words = set(edge_words + [query_word] + answer_words)
        word_to_id = {w: i for i, w in enumerate(sorted(all_words))}
        id_to_word = {i: w for w, i in word_to_id.items()}

        # 3. Reconstruct DAG: child ← [parents]
        dag = defaultdict(list)
        for i in range(0, len(edge_words), 2):
            p, c = edge_words[i], edge_words[i + 1]
            dag[word_to_id[c]].append(word_to_id[p])

        query = word_to_id[query_word]
        topo = [word_to_id[w] for w in answer_words]

        # 4. Get reachable nodes from query
        reachable = set()
        stack = [query]
        while stack:
            node = stack.pop()
            reachable.add(node)
            for parent in dag.get(node, []):
                if parent not in reachable:
                    stack.append(parent)

        if set(topo) != reachable:
            return False, query, [id_to_word[i] for i in topo]

        # 5. Validate topological order
        seen = set()
        for node in topo:
            for parent in dag.get(node, []):
                if parent not in seen:
                    return False, query, [id_to_word[i] for i in topo]
            seen.add(node)

        return True, query, [id_to_word[i] for i in topo]
    
    
rng = random.Random(42)


# NOTE: during testing, we enforce n = N and the code is NOT provided here

def topsort_data(N, multi=False):
    """按训练分布采样图规模，返回一条 Brevo 训练样本。"""
    _distribution_list_to_choose = list(range(3, N+1))
    power, bias = 1, pow(N, 0.5)
    p = [1.0 / (pow(i, power) + bias + 1e-12) for i in _distribution_list_to_choose]
    _distribution_p = [1.0 * x / sum(p) for x in p]
    n = rng.choices(_distribution_list_to_choose, weights=_distribution_p)[0]

    topo = TopoSortDepthStats(n, vocab_size=N)
    text, token_type, list_labels, depth = topo.generate_tokens(rng, multi=multi)
    
    ## token_type is for my own reference, not used for training
    ## list_labels = 0 or 1, where 1's are for answer tokens. We pretrain only on those answer tokens --- pretraining with all the tokens yield similar results, but is a factor slower.
    return {0:text, 1:token_type, 'label':list_labels}


bos_token_id = 9999
eos_token_id = 9998

print(topsort_data(N=110))  # used as our Brevo1 tasks
print(topsort_data(N=90))
print(topsort_data(N=70))

print(topsort_data(N=50,multi=True))  # used as our Brevo2 tasks
print(topsort_data(N=40,multi=True))
print(topsort_data(N=30,multi=True))




