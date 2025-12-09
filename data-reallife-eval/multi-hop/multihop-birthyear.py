# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
# This is a simple code to generate multi-hop in-context retrieval tasks about people's birth years.
# In the <Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers> paper as well as this repo, 
# I primarily used (N,k) = (5,1) and (3,2) for the 1-hop and 2-hop tasks respectively.
# I added some prompts for the k=2 case to improve evaluation performance.
#

class MultiHopTaskGenerator:
    def __init__(self, tokenizer):
        import nltk
        from nltk.tokenize import sent_tokenize
        nltk.download('punkt')
        nltk.download('punkt_tab')
        self.tokenizer = tokenizer
        
        self.background_sentences = []
        wiki_file = "<zeyuan_place_holder>/wikibook/loose_wiki_valid.json"   # I used some wikipedia test data as the background "junk" text.
        with open(wiki_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                paragraph = data.get("plaintext", "")
                sentences = sent_tokenize(paragraph)
                self.background_sentences.extend([s.strip() for s in sentences if s.strip()])
        
        self.background_token_lengths = []
        for sentence in self.background_sentences:
            token_ids = self.tokenizer(sentence, add_special_tokens=False)["input_ids"]
            self.background_token_lengths.append(len(token_ids))
        
        self.first_names = self._load_names(os.path.join(get_base_folder(), "first_name.txt"))     # I used the same files from the Capo (bioS/bioR) datasets, also released in this repo.
        self.middle_names = self._load_names(os.path.join(get_base_folder(), "middle_name.txt"))   # I used the same files from the Capo (bioS/bioR) datasets, also released in this repo.
        self.last_names = self._load_names(os.path.join(get_base_folder(), "last_name.txt"))       # I used the same files from the Capo (bioS/bioR) datasets, also released in this repo.
        
        self.random_obj = random.Random(42)  # I used a fixed seed to ensure fairness across different model evaluations.

    def reset_randomness(self):
        self.random_obj = random.Random(42)
    
    def _load_names(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _generate_unique_full_names(self, total_count):
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
        total_people = N * k
        unique_names = self._generate_unique_full_names(total_people)
        
        sentences = []
        age_range = list(range(11, 30))
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
            self.random_obj.shuffle(indices)
            for j in range(N):
                name_current = unique_names[start_index + j]
                mapped_index = indices[j]
                mapped_person = previous_group[mapped_index]
                age = mapped_person[1]
                s = f"{name_current} was born in the same year as {mapped_person[0]}." # 
                current_group.append((name_current, age))
                sentences.append(s)
            previous_group = current_group
        
        chosen_person = previous_group[0]
        question_prefix = f"\n\nAnswer me: {chosen_person[0]} was born in the year of "
        question_sentence = question_prefix + f"{chosen_person[1]}."
        sentences.append(question_sentence)
        
        return sentences, chosen_person, question_prefix
    
    def _select_background(self, M):
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
        multi_hop_sentences, chosen_person, question_prefix = self._generate_multi_hop_task(N, k)
        problem_sentences = multi_hop_sentences[:-1]
        
        answer_str = str(chosen_person[1])
        
        problem_sentences = [s if s.endswith('.') else s + '.' for s in problem_sentences]
        if not question_prefix.endswith(' '):
            question_prefix += ' '
        
        background = self._select_background(M)
        
        interleaved = self._interleave(background, problem_sentences)
        full_text = " ".join(interleaved)
        
        if k==1:
            full_text = full_text + " " + question_prefix
        else:
            few_shot = "You will be asked questions about people's birth years, and the birth year descriptions are hidden in some random text. Some people's birth years are directly given, while others are given in the form that `name1' was born in the same year as `name2'.\n\n"
            full_text = few_shot + "<context>" + full_text + "</context>" + question_prefix

        full_text_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        return full_text_ids, answer_str