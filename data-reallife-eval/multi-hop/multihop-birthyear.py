# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
# This is a simple code to generate multi-hop in-context retrieval tasks about people's birth years.
# In the <Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers> paper as well as this repo, 
# I primarily used (N,k) = (5,1) and (3,2) for the 1-hop and 2-hop tasks respectively.
# I added some prompts for the k=2 case to improve evaluation performance.
#
"""
本文件实现“生日年份多跳检索”任务样本生成器。

任务直观描述：
- 先生成一批“某人出生于某年”的基础事实；
- 再生成若干层“X 与 Y 同年出生”的链式关系（形成多跳）；
- 把这些事实与背景干扰句子混合，最终拼成长上下文；
- 在末尾给出问题前缀，要求模型补全具体年份。

关键超参数：
- N: 每一层的人数（同层节点数）；
- k: 跳数/层数（k=1 表示直接检索，k=2 表示两跳推理）；
- M: 背景干扰文本预算（按 token 长度近似控制）。
"""

class MultiHopTaskGenerator:
    """生成多跳出生年份问答样本的工具类。"""

    def __init__(self, tokenizer):
        # 仅在初始化时导入并下载分句资源，避免模块导入阶段的额外副作用。
        import nltk
        from nltk.tokenize import sent_tokenize
        nltk.download('punkt')
        nltk.download('punkt_tab')
        # 外部传入 tokenizer，用于：
        # 1) 估算背景句 token 长度；
        # 2) 将最终文本编码为 token id 序列。
        self.tokenizer = tokenizer
        
        # 读取并缓存背景“干扰句”语料。
        self.background_sentences = []
        wiki_file = "<zeyuan_place_holder>/wikibook/loose_wiki_valid.json"   # I used some wikipedia test data as the background "junk" text.
        with open(wiki_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                paragraph = data.get("plaintext", "")
                # 将段落切分成句子，后续按句子粒度随机抽取。
                sentences = sent_tokenize(paragraph)
                self.background_sentences.extend([s.strip() for s in sentences if s.strip()])
        
        # 预计算每个背景句的 token 长度，避免生成样本时重复编码。
        self.background_token_lengths = []
        for sentence in self.background_sentences:
            token_ids = self.tokenizer(sentence, add_special_tokens=False)["input_ids"]
            self.background_token_lengths.append(len(token_ids))
        
        # 读取姓名词表，用于随机拼接唯一全名。
        self.first_names = self._load_names(os.path.join(get_base_folder(), "first_name.txt"))     # I used the same files from the Capo (bioS/bioR) datasets, also released in this repo.
        self.middle_names = self._load_names(os.path.join(get_base_folder(), "middle_name.txt"))   # I used the same files from the Capo (bioS/bioR) datasets, also released in this repo.
        self.last_names = self._load_names(os.path.join(get_base_folder(), "last_name.txt"))       # I used the same files from the Capo (bioS/bioR) datasets, also released in this repo.
        
        # 固定随机种子，保证不同模型评测时数据一致（可复现/可公平对比）。
        self.random_obj = random.Random(42)  # I used a fixed seed to ensure fairness across different model evaluations.

    def reset_randomness(self):
        # 重置随机状态，复现实验序列。
        self.random_obj = random.Random(42)
    
    def _load_names(self, file_path):
        # 从词表文件读取非空姓名片段。
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _generate_unique_full_names(self, total_count):
        # 生成 total_count 个不重复全名，形式为「名 中间名 姓」。
        names_set = set()
        full_names = []
        while len(full_names) < total_count:
            f_name = self.random_obj.choice(self.first_names)
            m_name = self.random_obj.choice(self.middle_names)
            l_name = self.random_obj.choice(self.last_names)
            full_name = f"{f_name} {m_name} {l_name}"
            if full_name not in names_set:
                names_set.add(full_name)
                full_names.append(full_name)
        return full_names
    
    def _generate_multi_hop_task(self, N, k):
        # 生成核心多跳事实：
        # - 第 1 层：直接给出生年；
        # - 第 2~k 层：通过“同年出生”映射上一层，形成链式检索。
        total_people = N * k
        unique_names = self._generate_unique_full_names(total_people)
        
        sentences = []
        age_range = list(range(11, 30))
        # 实际采用年份范围（覆盖 1950~2009）。
        age_range = list(range(1950, 2010))
        group1_ages = self.random_obj.sample(age_range, N)
        group1 = []
        for i in range(N):
            name = unique_names[i]
            age = group1_ages[i]
            s = f"{name} was born in the year of {age}."
            group1.append((name, age))
            sentences.append(s)
        
        previous_group = group1
        for group_index in range(2, k + 1):
            current_group = []
            start_index = (group_index - 1) * N
            indices = list(range(N))
            # 打乱映射，避免固定位置对应导致模式过强。
            self.random_obj.shuffle(indices)
            for j in range(N):
                name_current = unique_names[start_index + j]
                mapped_index = indices[j]
                mapped_person = previous_group[mapped_index]
                age = mapped_person[1]
                # 当前层仅通过“同年”关系间接获得年份信息。
                s = f"{name_current} was born in the same year as {mapped_person[0]}." # 
                current_group.append((name_current, age))
                sentences.append(s)
            previous_group = current_group
        
        # 默认选最后一层第一个人作为问题目标。
        chosen_person = previous_group[0]
        # 问题前缀以补全式形式出现，便于评测生成结果。
        question_prefix = f"\n\nAnswer me: {chosen_person[0]} was born in the year of "
        question_sentence = question_prefix + f"{chosen_person[1]}."
        sentences.append(question_sentence)
        
        return sentences, chosen_person, question_prefix
    
    def _select_background(self, M):
        # 按 token 预算 M 选择连续背景句作为干扰文本。
        if M <= 0:
            return []
        total_sentences = len(self.background_sentences)
        start_index = self.random_obj.randint(0, total_sentences - 1)
        selected = []
        total_length = 0
        for i in range(start_index, total_sentences):
            sentence_length = self.background_token_lengths[i]
            if total_length + sentence_length < M:
                selected.append(self.background_sentences[i])
                total_length += sentence_length
            else:
                break
        return selected
    
    def _interleave(self, list1, list2):
            # 将两组句子随机交错：list1 通常为背景句，list2 为任务事实句。
        total_length = len(list1) + len(list2)
        positions_for_list2 = set(self.random_obj.sample(range(total_length), len(list2)))
        interleaved = []
        i1 = 0
        i2 = 0
        for pos in range(total_length):
            if pos in positions_for_list2:
                interleaved.append(list2[i2])
                i2 += 1
            else:
                interleaved.append(list1[i1])
                i1 += 1
        return interleaved
    
    def generate_sample(self, N, k, M):
        """
        生成一条样本并返回模型输入 token ids 与标准答案字符串。

        参数：
        - N: 每层人数；
        - k: 跳数（层数）；
        - M: 背景干扰 token 预算。

        返回：
        - full_text_ids: 最终输入文本的 token id 列表；
        - answer_str: 目标人物出生年份（字符串形式）。
        """
        multi_hop_sentences, chosen_person, question_prefix = self._generate_multi_hop_task(N, k)
        problem_sentences = multi_hop_sentences[:-1]
        
        answer_str = str(chosen_person[1])
        
        # 保证事实句以句号结束，减少格式噪声。
        problem_sentences = [s if s.endswith('.') else s + '.' for s in problem_sentences]
        if not question_prefix.endswith(' '):
            question_prefix += ' '
        
        background = self._select_background(M)
        
        interleaved = self._interleave(background, problem_sentences)
        full_text = " ".join(interleaved)
        
        if k==1:
            # 一跳任务：直接在长上下文后接问题前缀。
            full_text = full_text + " " + question_prefix
        else:
            # 多跳任务：前置简短说明 + <context> 包裹，帮助模型理解“同年映射”规则。
            few_shot = "You will be asked questions about people's birth years, and the birth year descriptions are hidden in some random text. Some people's birth years are directly given, while others are given in the form that `name1' was born in the same year as `name2'.\n\n"
            full_text = few_shot + "<context>" + full_text + "</context>" + question_prefix

        # 输出 tokenizer 后的 id 序列，供后续模型推理与评测。
        full_text_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        return full_text_ids, answer_str