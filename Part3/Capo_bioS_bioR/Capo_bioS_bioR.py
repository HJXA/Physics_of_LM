# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
# This gives the necessary code to generate our [bioS]/[bioR] datasets used in [Physics of language models: Part 3.1, 3.2, 3.3], 
#      in which the [bioS] dataset was also used in [Physics of language models: Part 4.1] as Task [Capo].
#
"""
Capo（bioS/bioR）数据生成与增强脚本。

文件包含三个关键函数：
1) generate_prompt2: 构造给 Llama 生成人物传记的提示词（用于 bioR）；
2) get_text_simple3: 直接合成结构化传记文本（用于 bioS）；
3) augmentation_permutation2: 对传记句子做重排与代词/姓名修复，增强知识位置鲁棒性。

说明：该文件以“模板池 + 随机采样 + 后处理修复”为主，不涉及训练逻辑。
"""
import random
import numpy as np


###################################################################################################################
# The following code was used in [Physics of language models: Part 3.1, 3.2, 3.3] to generate the prompts 
# for Llama1/2 to generate synthetic biographies for the [bioR] dataset
# mode=0 was our prompt for Llama1 and mode=1 was for Llama2
###################################################################################################################
def generate_prompt2(self, word=None, mode=0):
    """生成用于大模型写人物传记的英文提示词。

    参数：
    - word: 期望词数（可空）；
    - mode: 0 表示旧版提示模板，1 表示 Llama2 模板。

    该函数会把人物关键事实（学校、专业、出生地、公司、生日）
    以随机句式拼接进 prompt，提升生成多样性。
    """
    if mode==1: # for llama2
        prompt = f"Write a {word}-word biography"
        prompt += f" about a person whose name is {self.first_name} {self.middle_name} {self.last_name}."
        prompt += " Be creative in the writing and permute sentences as much as you can."
    else:
        assert mode==0
        prompt = f"Write a {np.random.randint(50, 200)} words biography"
        if word is not None:
            prompt = f"Write a {word} words biography"
            #prompt = f"Rewrite a {word} words biography in French"

        prompt += f" about a person whose name is {self.first_name} {self.middle_name} {self.last_name}."
    
    prompts = []
    sentences = [
    " The person studied at {university}.",
    " The person attended {university} for education.",
    " The person completed studies at {university}.",
    " The person received degree from {university}.",
    " The person graduated from {university}.",
    " The person was enrolled at {university} for studies.",
    " The person completed education at {university}.",
    ]
    prompts += [random.choice(sentences).format(university=self.university)]

    sentences = [
    " The person studied {field} there.",
    " The person majored in {field} there.",
    " The person pursued a degree in {field} there.",
    " The person received a degree in {field} there."
    ]
    prompts[-1] += random.choice(sentences).format(field=self.field)

    sentences = [
    " The person was born in {birthcity}.",
    " The person originated from {birthcity}.",
    " The person is a native of {birthcity}.",
    " The person calls {birthcity} birthplace.",
    " The person was brought up in {birthcity}.",
    " The person was born and raised in {birthcity}.",
    ]
    prompts += [random.choice(sentences).format(birthcity=self.birthcity)]

    sentences = [
    " The person worked in {company1city} for {company1name}.",
    " The person was employed by {company1name} in {company1city}.",
    " The person had a job at {company1name} located in {company1city}.",
    " The person spent time working for {company1name} in the city of {company1city}.",
    " The person had a professional experience at {company1name} headquarters in {company1city}.",
    " The person played a vital role at {company1name} while living in {company1city}.",
    " The person enjoyed a rewarding career at {company1name} based in {company1city}.",
    " In {company1city}, the person was part of the team at {company1name}."
    " The person's professional journey included working in {company1city} for {company1name}.",
    " The person served as an employee at {company1name} based in {company1city}.",
    " In {company1city}, the person joined the team at {company1name}.",
    " The person's professional tenure included {company1city}, where the person worked for {company1name}.",
    " In {company1city}, the person played a key role in {company1name} as an employee.",
    " In {company1city}, the person's employment history included working at {company1name}."
    ]
    prompts += [random.choice(sentences).format(company1city=self.company1city, company1name=self.company1name)]

    sentences = [
    " The person was born on {birthday}.",
    " The person's birthday falls on {birthday}.",
    ]
    prompts += [random.choice(sentences).format(birthday=f"{self.birthmonth} {self.birthday}, {self.birthyear}")]
    random.shuffle(prompts)
    prompt += "".join(prompts)

    return prompt



###################################################################################################################
# The following code was used in [Physics of language models: Part 3.1, 3.2, 3.3] to generate the 
#   fully synthetic [bioS] dataset without knowledge augmentation (e.g., permutation).
#   It's easy to adapt this code to perform augmentation.
#   but you can also use our augmentation_permutation2 method below.
###################################################################################################################
def get_text_simple3(person, order=0, reverse_md = False, fixed_templates=False):
    """根据结构化 `person` 信息直接生成一段合成传记文本。

    - order 控制公司城市/公司名称两条信息的先后顺序；
    - reverse_md 控制生日中的“月-日”排列格式。
    - fixed_templates=True 时固定使用每个模板池的第一个句式。

    函数内部维护多组同义句模板，通过随机采样降低表述单一性。
    """
    sentence_structures1 = [
        "{name} was born on {birthday}.",
        "{name}'s birthday falls on {birthday}.",
        "{name} celebrates their birthday on {birthday}.",
        "{name} came into this world on {birthday}.",
        "{name}'s birth date is {birthday}.",
        "{name} arrived on {birthday}.",
        "{name} entered the world on {birthday}.",
        "{name} was brought into existence on {birthday}.",
        "{name} took their first breath on {birthday}.",
        "{name} celebrates their special day on {birthday}.",
        "{name} marks their birthday every year on {birthday}.",
        "{name} honors their birth day on {birthday}.",
        "{name} was born on the memorable date of {birthday}.",
        "{name} was gifted to the world on {birthday}.",
        "{name} has their annual celebration on {birthday}.",
        "{name} celebrates another year of life on {birthday}.",
        "{name} commemorates their birth anniversary on {birthday}.",
        "{name} entered the world with joy on {birthday}.",
        "{name} was born into this beautiful world on {birthday}.",
        "{name} came into existence on the significant date of {birthday}.",
        "{name} arrived on this Earth on {birthday}.",
        "{name} celebrates their special day each year on {birthday}.",
        "{name} recognizes {birthday} as their birth date.",
        "{name} looks forward to their birthday every year on {birthday}.",
        "{name} pays tribute to the day they were born, {birthday}.",
        "{name} celebrates their birth on the remarkable day of {birthday}.",
        "{name} arrived in this world on {birthday}, a day to be remembered.",
        "{name} was born on the auspicious day of {birthday}.",
        "{name}'s birth is celebrated annually on {birthday}.",
        "{name} commemorates their birth on the same day each year, {birthday}.",
        "{name} celebrates their life on the day of {birthday}.",
        "{name} acknowledges their birth day as {birthday}.",
        "{name} rejoices on {birthday}, the day they were born.",
        "{name} reflects on their birth day, {birthday}, with gratitude.",
        "{name} celebrates their special day of {birthday} every year.",
        "{name} was born on {birthday}, a day that holds significance in their life.",
        "{name} marks {birthday} as the day they began their journey.",
        "{name} arrived in this world with joy and blessings on {birthday}.",
        "{name} pays tribute to their birth day, {birthday}, each year.",
        "{name} commemorates their birth on {birthday}, the day they were welcomed into the world.",
        "{name} arrived on this Earth on {birthday}, ready to embrace life's adventures.",
        "{name} celebrates the anniversary of their birth on {birthday}.",
        "{name} acknowledges {birthday} as the day they were born.",
        "{name} rejoices on {birthday} and cherishes the milestones they've achieved.",
        "{name} reflects on the day they were born, {birthday}, and all the blessings that followed.",
        "{name} celebrates their life journey every year on {birthday}."
    ]

    sentence_structures2 = [
        "{name} was born in {birthcity}.",
        "{name} hails from {birthcity}.",
        "{name} originated from {birthcity}.",
        "{name} is a native of {birthcity}.",
        "{name} came into the world in {birthcity}.",
        "{name} first saw the light of day in {birthcity}.",
        "{name} entered this world in {birthcity}.",
        "{name} took their first breath in {birthcity}.",
        "{name} was brought into existence in {birthcity}.",
        "{name} started their life journey in {birthcity}.",
        "{name} calls {birthcity} their birthplace.",
        "{name} has roots in {birthcity}.",
        "{name} has a deep connection to {birthcity}.",
        "{name} owes their birth to {birthcity}.",
        "{name} traces their origins back to {birthcity}.",
        "{name} has sentimental ties to {birthcity}.",
        "{name} has fond memories of {birthcity}.",
        "{name} has a special bond with {birthcity}.",
        "{name} proudly identifies as a native of {birthcity}.",
        "{name} holds {birthcity} close to their heart.",
        "{name} cherishes their connection to {birthcity}.",
        "{name} was brought up in {birthcity}.",
        "{name} spent their early years in {birthcity}.",
        "{name} has vivid recollections of {birthcity}.",
        "{name} has a strong sense of belonging to {birthcity}.",
        "{name} often reminisces about {birthcity}.",
        "{name} has family ties to {birthcity}.",
        "{name} owes their heritage to {birthcity}.",
        "{name} associates their identity with {birthcity}.",
        "{name} has deep cultural roots in {birthcity}.",
        "{name} embraces their birth city of {birthcity}.",
        "{name} takes pride in their birthplace, {birthcity}.",
        "{name} was welcomed into the world in {birthcity}.",
        "{name} has a strong affinity for {birthcity}.",
        "{name} reminisces about their early days in {birthcity}.",
        "{name} has a personal connection to {birthcity}.",
        "{name} has a deep sense of nostalgia for {birthcity}.",
        "{name} was born and raised in {birthcity}.",
        "{name} owes their roots to {birthcity}.",
        "{name} holds a special place in their heart for {birthcity}.",
        "{name} has a unique bond with {birthcity}.",
        "{name} was birthed in the beautiful city of {birthcity}.",
        "{name} has a profound appreciation for {birthcity}.",
        "{name} associates their childhood with {birthcity}.",
        "{name} always carries a piece of {birthcity} within them.",
        "{name} reflects on their upbringing in {birthcity}.",
        "{name} has a strong attachment to {birthcity}.",
        "{name} celebrates their birth in {birthcity}.",
        "{name} feels a deep connection to {birthcity}."
    ]
    sentence_structures3 = [
        "{name} studied at {university}.",
        "{name} attended {university} for their education.",
        "{name} completed their studies at {university}.",
        "{name} received their degree from {university}.",
        "{name} pursued their education at {university}.",
        "{name} graduated from {university}.",
        "{name} earned their degree at {university}.",
        "{name} obtained their diploma from {university}.",
        "{name} was enrolled at {university} for their studies.",
        "{name} undertook their academic journey at {university}.",
        "{name} completed their education at {university} with distinction.",
        "{name} specialized in their field of study at {university}.",
        "{name} acquired their knowledge and skills at {university}.",
        "{name} pursued advanced coursework at {university}.",
        "{name} engaged in research projects while studying at {university}.",
        "{name} was an active member of the academic community at {university}.",
        "{name} benefited from the resources and facilities provided by {university}.",
        "{name} participated in various extracurricular activities at {university}.",
        "{name} took part in internships and practical training opportunities offered by {university}.",
        "{name} was mentored by distinguished professors at {university}.",
        "{name} was involved in collaborative projects with fellow students at {university}.",
        "{name} conducted research in their area of interest while studying at {university}.",
        "{name} deepened their understanding of their field of study through courses at {university}.",
        "{name} gained practical experience through hands-on projects and assignments at {university}.",
        "{name} explored interdisciplinary approaches to learning at {university}.",
        "{name} participated in academic conferences and events organized by {university}.",
        "{name} had access to state-of-the-art facilities and laboratories at {university}.",
        "{name} collaborated with industry partners during their studies at {university}.",
        "{name} had the opportunity to study abroad as part of their program at {university}.",
        "{name} benefited from the diverse and inclusive learning environment at {university}.",
        "{name} was recognized for their academic achievements at {university}.",
        "{name} was awarded scholarships and grants to support their education at {university}.",
        "{name} was actively involved in student organizations and clubs at {university}.",
        "{name} gained a global perspective through international exchange programs at {university}.",
        "{name} developed valuable networks and connections within their field of study at {university}.",
        "{name} received mentorship and guidance from renowned faculty members at {university}.",
        "{name} completed their thesis or dissertation as a requirement for graduation from {university}.",
        "{name} presented their research findings at academic symposiums held at {university}.",
        "{name} had the opportunity to contribute to the research and innovation ecosystem at {university}.",
        "{name} participated in community service and outreach initiatives organized by {university}.",
        "{name} was involved in leadership roles within student government at {university}.",
        "{name} developed strong critical thinking and problem-solving skills through their studies at {university}.",
        "{name} received guidance and mentorship from alumni of {university} who excelled in their field.",
        "{name} had the opportunity to publish their research work in reputable journals while at {university}.",
        "{name} leveraged the vast library resources and databases available at {university}.",
        "{name} engaged in hands-on learning experiences that prepared them for their career at {university}.",
        "{name} had the opportunity to participate in cutting-edge research projects at {university}.",
        "{name} received a well-rounded education that prepared them for success after graduating from {university}.",
        "{name} was part of a vibrant and diverse student community at {university}.",
    ]

    sentence_structures4 = [
        "{name} studied {field}.",
        "{name} majored in {field}.",
        "{name} pursued a degree in {field}.",
        "{name} specialized in {field}.",
        "{name} focused on {field} during their studies.",
        "{name} has in-depth knowledge of {field}.",
        "{name} gained expertise in {field}.",
        "{name} acquired skills in {field}.",
        "{name} completed their education with a focus on {field}.",
        "{name} has a strong background in {field}.",
        "{name} dedicated their studies to {field}.",
        "{name} excelled in {field}.",
        "{name} deepened their understanding of {field}.",
        "{name} specialized in the field of {field}.",
        "{name} pursued advanced studies in {field}.",
        "{name} conducted research in {field}.",
        "{name} explored the various aspects of {field}.",
        "{name} gained practical experience in {field}.",
        "{name} analyzed {field} in their studies.",
        "{name} developed a strong foundation in {field}.",
        "{name} applied their knowledge of {field}.",
        "{name} completed a comprehensive program in {field}.",
        "{name} was recognized for their work in {field}.",
        "{name} specialized in {field} with a focus on practical applications.",
        "{name} pursued advanced coursework in {field}.",
        "{name} conducted experiments in {field}.",
        "{name} researched innovative approaches in {field}.",
        "{name} gained hands-on experience in {field}.",
        "{name} explored the theoretical aspects of {field}.",
        "{name} deepened their understanding of {field} through coursework.",
        "{name} applied their knowledge of {field} to real-world problems.",
        "{name} specialized in {field} and its related disciplines.",
        "{name} engaged in collaborative projects in {field}.",
        "{name} developed a strong theoretical foundation in {field}.",
        "{name} acquired practical skills relevant to {field}.",
        "{name} conducted in-depth research in {field}.",
        "{name} explored emerging trends in {field}.",
        "{name} gained expertise in the field of {field} through hands-on projects.",
        "{name} completed a rigorous program in {field}.",
        "{name} was actively involved in {field} research.",
        "{name} participated in internships related to {field}.",
        "{name} studied the principles of {field} extensively.",
        "{name} acquired a deep understanding of {field} concepts.",
        "{name} specialized in {field} and its applications.",
        "{name} pursued interdisciplinary studies related to {field}.",
        "{name} gained practical knowledge in {field} through real-world projects.",
        "{name} explored the intersection of {field} and technology.",
        "{name} conducted fieldwork in {field}.",
        "{name} gained insights into {field} through hands-on experiments.",
        "{name} studied {field} and its impact on society.",
        "{name} acquired practical skills applicable to {field}.",
        "{name} conducted research on cutting-edge {field} topics."
    ]


    sentence_structures5 = [
        "{name} worked in {company1city}.",
        "{name} had a job in {company1city}.",
        "{name} was employed in {company1city}.",
        "{name} spent time working in {company1city}.",
        "{name} was part of the workforce in {company1city}.",
        "{name} had a professional role in {company1city}.",
        "{name} had a job opportunity in {company1city}.",
        "{name} contributed to the economy of {company1city}.",
        "{name} gained work experience in {company1city}.",
        "{name} was employed at a company based in {company1city}.",
        "{name} joined the workforce in {company1city}.",
        "{name} was part of a professional team in {company1city}.",
        "{name} was engaged in work activities in {company1city}.",
        "{name} developed their career in {company1city}.",
        "{name} had employment prospects in {company1city}.",
        "{name} worked for a company located in {company1city}.",
        "{name} played a role in the business sector of {company1city}.",
        "{name} held a position in {company1city}.",
        "{name} contributed to the success of a company in {company1city}.",
        "{name} pursued professional opportunities in {company1city}.",
        "{name} was involved in the industry of {company1city}.",
        "{name} gained valuable skills while working in {company1city}.",
        "{name} made professional connections in {company1city}.",
        "{name} experienced the work culture of {company1city}.",
        "{name} was part of a dynamic work environment in {company1city}.",
        "{name} contributed to the growth of a company in {company1city}.",
        "{name} worked on projects in {company1city}.",
        "{name} was employed by a reputable company in {company1city}.",
        "{name} acquired industry knowledge while working in {company1city}.",
        "{name} collaborated with colleagues in {company1city}.",
        "{name} was immersed in the professional scene of {company1city}.",
        "{name} contributed their expertise to a company in {company1city}.",
        "{name} gained insights into the business landscape of {company1city}.",
        "{name} worked with clients and customers from {company1city}.",
        "{name} participated in projects that impacted {company1city}.",
        "{name} was part of the workforce driving innovation in {company1city}.",
        "{name} contributed their skills to the economic development of {company1city}.",
        "{name} worked in {company1city} and made a positive impact in their field.",
        "{name} was employed by a leading company in {company1city}.",
        "{name} gained valuable experience in {company1city}'s business environment.",
        "{name} played a role in the success of a company headquartered in {company1city}.",
        "{name} was involved in the professional community of {company1city}.",
        "{name} contributed to the local economy of {company1city}.",
        "{name} worked with diverse colleagues in {company1city}.",
        "{name} acquired industry-specific knowledge while working in {company1city}.",
        "{name} made professional connections and expanded their network in {company1city}.",
        "{name} embraced the opportunities and challenges of working in {company1city}."
    ]

    sentence_structures6 = [
        "{name} worked at {company1name}.",
        "{name} was employed by {company1name}.",
        "{name} had a job at {company1name}.",
        "{name} spent time working at {company1name}.",
        "{name} was part of the team at {company1name}.",
        "{name} had a professional role at {company1name}.",
        "{name} had a job opportunity at {company1name}.",
        "{name} contributed to the success of {company1name}.",
        "{name} gained work experience at {company1name}.",
        "{name} was employed by the renowned {company1name}.",
        "{name} joined {company1name} as an employee.",
        "{name} was part of the workforce at {company1name}.",
        "{name} was engaged in work activities at {company1name}.",
        "{name} developed their career at {company1name}.",
        "{name} had employment prospects at {company1name}.",
        "{name} worked for {company1name}, a leading company.",
        "{name} played a role in {company1name}'s operations.",
        "{name} held a position at {company1name}.",
        "{name} contributed to the growth of {company1name}.",
        "{name} pursued professional opportunities at {company1name}.",
        "{name} gained valuable skills while working at {company1name}.",
        "{name} made professional connections at {company1name}.",
        "{name} experienced the work culture at {company1name}.",
        "{name} was part of a dynamic work environment at {company1name}.",
        "{name} contributed to the success of {company1name} in their role.",
        "{name} worked on projects at {company1name}.",
        "{name} was employed at {company1name}, a respected company.",
        "{name} acquired industry knowledge while working at {company1name}.",
        "{name} collaborated with colleagues at {company1name}.",
        "{name} was immersed in the professional scene at {company1name}.",
        "{name} contributed their expertise to {company1name}.",
        "{name} gained insights into the industry while working at {company1name}.",
        "{name} worked with clients and customers of {company1name}.",
        "{name} participated in projects that impacted {company1name}.",
        "{name} was part of the workforce driving innovation at {company1name}.",
        "{name} contributed their skills to the success of {company1name}.",
        "{name} worked at {company1name} and made a positive impact in their field.",
        "{name} was employed by {company1name}, a reputable company.",
        "{name} gained valuable experience at {company1name} in their role.",
        "{name} played a role in the success of {company1name}.",
        "{name} was involved in the day-to-day operations of {company1name}.",
        "{name} was an integral part of {company1name}'s team.",
        "{name} contributed to the growth and development of {company1name}.",
        "{name} made significant contributions to {company1name} during their tenure.",
        "{name} embraced the opportunities and challenges of working at {company1name}.",
        "{name} was a key asset to {company1name}'s success.",
        "{name} contributed to the achievements and milestones of {company1name}.",
        "{name} worked diligently at {company1name} to achieve their goals."
    ]
    
    pick = (lambda structures: structures[0]) if fixed_templates else random.choice

    sentence1 = " " + pick(sentence_structures1)
    sentence2 = " " + pick(sentence_structures2)
    sentence3 = " " + pick(sentence_structures3)
    sentence4 = " " + pick(sentence_structures4)
    sentence5 = " " + pick(sentence_structures5)
    sentence6 = " " + pick(sentence_structures6)
    name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    he_she = 'He' if person['id']%2==0 else 'She'
    if reverse_md:
        ans = sentence1.format(name=name, birthday=f"{person['birthday']} of {person['birthmonth']}, {person['birthyear']}")
    else:
        ans = sentence1.format(name=name, birthday=f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}")
    ans += sentence2.format(name=he_she, birthcity=f"{person['birthcity']}")
    ans += sentence3.format(name=he_she, university=f"{person['university']}")
    ans += sentence4.format(name=he_she, field=f"{person['field']}")
    if order==0:
        ans += sentence5.format(name=he_she, company1city=f"{person['company1city']}")
        ans += sentence6.format(name=he_she, company1name=f"{person['company1name']}")
    else:
        ans += sentence6.format(name=he_she, company1name=f"{person['company1name']}")
        ans += sentence5.format(name=he_she, company1city=f"{person['company1city']}")
    return ans







###################################################################################################################
# The following code was used in [Physics of language models: Part 3.1, 3.2, 3.3] to perform 
#   knowledge augmentation (permutation) on a given biography. While it is easy to do so for [bioS] data, for
#   Llama1/2-generated biographies, we wrote this code to handle various real-life cases. 
# Note this code may not always succeed, sometimes Llama1/2-generated biographies may not even contain the person's
#   full name and there are numerous edge cases; but we manually checked and such edge cases are sufficiently rare.
###################################################################################################################
def augmentation_permutation2(person, text):
    """对传记做句子级重排，并修复首句实体指代。

    主要流程：
    1) 预清洗文本与特殊缩写保护（避免句点切分错误）；
    2) 识别原文中的人物指代形式（The person / He / She / They / 姓名片段）；
    3) 句子乱序后，强制在首句恢复完整姓名；
    4) 其余句子按需要回退到代词形式；
    5) 恢复特殊缩写中的句点。
    """

    text = text.replace('(50 words)','')
    text = text.replace('\n',' ').replace('  ',' ').replace('  ',' ')
    text = text.strip()
    found_the_person = 'The person' in text or 'the person' in text
    found_she = False
    found_he = False
    found_first = False
    found_last = False
    found_middle = False
    found_they = False
    nname = f"{person['first_name']} {person['middle_name']} {person['last_name']}"

    if not found_the_person:
        found_she = ' She ' in text or ' she ' in text
        if not found_she:
            found_he = ' He ' in text or ' he ' in text
            if not found_he:
                found_they = ' They ' in text or ' they ' in text
    found_first = text.count(person['first_name'])>=3
    if not found_first:
        found_last = text.count(person['last_name'])>=3
        if not found_last:
            found_middle = text.count(person['middle_name'])>=3
            if not found_middle:
                found_first = text.count(person['first_name'])>=2
    if not found_the_person and not found_she and not found_he and not found_first and not found_last and not found_middle and not found_they:
        print('---')
        print(f"DID NOT FIND HE/SHE/NAME\nOLD:{text}")
        print()
    ori_text = text
    spe_period = ["C.H. Robinson", "Caterpillar Inc.", "Dow Inc.", "St. Louis", " D.C.", "St. Petersburg", "Port St. Lucie", " Inc.", "Steven O. Rice", " Dr.", "Ph.D.", "B.S.", "Skelton, A. R.", "U.S."]
    for x in spe_period:
        y = x.replace('.','#')
        text = text.replace(x,y)
    if text[-1]!='.':
        text = text + '.'

    if found_first:
        text = text.replace(nname, person['first_name'])
    if found_last:
        text = text.replace(nname, person['last_name'])
    if found_middle:
        text = text.replace(nname, person['middle_name'])
    #assert text[-1]=='.', 's' + text + 'e'
    text = text[:-1].split('. ')

    text_bak = text.copy()
    for times in range(1000):
        text = text_bak.copy()
        random.shuffle(text)

        text[0] = ' ' + text[0]
        if found_first:
            text[0] = text[0].replace(f" {person['first_name']}", f' {nname}')
        if found_last:
            text[0] = text[0].replace(f" {person['last_name']}", f' {nname}')
        if found_middle:
            text[0] = text[0].replace(f" {person['middle_name']}", f' {nname}')
        if found_the_person:
            if ' The person ' in text[0]:
                text[0] = text[0].replace(' The person ', f" {person['first_name']} {person['middle_name']} {person['last_name']} ", 1)
            else:
                text[0] = text[0].replace(' the person ', f" {person['first_name']} {person['middle_name']} {person['last_name']} ", 1)
        if found_she:
            if ' She ' in text[0]:
                text[0] = text[0].replace(' She ', f" {person['first_name']} {person['middle_name']} {person['last_name']} ", 1)
            else:
                text[0] = text[0].replace(' she ', f" {person['first_name']} {person['middle_name']} {person['last_name']} ", 1)
        if found_he:
            if ' He ' in text[0]:
                text[0] = text[0].replace(' He ', f" {person['first_name']} {person['middle_name']} {person['last_name']} ", 1)
            else:
                text[0] = text[0].replace(' he ', f" {person['first_name']} {person['middle_name']} {person['last_name']} ", 1)
        if found_they:
            if ' They ' in text[0]:
                text[0] = text[0].replace(' They ', f" {person['first_name']} {person['middle_name']} {person['last_name']} ", 1)
            else:
                text[0] = text[0].replace(' they ', f" {person['first_name']} {person['middle_name']} {person['last_name']} ", 1)
        full_name_not_found = False
        if nname not in text[0]:
            text[0] = text[0].replace(f" {person['first_name']} {person['last_name']} ", f" {nname} ", 1)
            if nname not in text[0]:
                text[0] = text[0].replace(f" {person['middle_name']} {person['last_name']} ", f" {nname} ", 1)
                if nname not in text[0]:
                    text[0] = text[0].replace(f" {person['first_name']} {person['middle_name']} ", f" {nname} ", 1)
                    if nname not in text[0]:
                        text[0] = text[0].replace(f" {person['first_name']} ", f" {nname} ", 1)
                        if nname not in text[0]:
                            text[0] = text[0].replace(f" {person['last_name']} ", f" {nname} ", 1)
                            if nname not in text[0]:
                                text[0] = text[0].replace(f" {person['middle_name']} ", f" {nname} ", 1)
                                if nname not in text[0]:
                                    full_name_not_found = True
        if full_name_not_found:
            if times==999:
                print('---')
                print(f"AFTER PERMUTE DID NOT FIND FULL NAME\nOLD:{text_bak}")
                print(f"NEW:{text}")
                print(f"name='{nname}' and found_he = {found_he} and found_she = {found_she} and found_they = {found_they} and found_first={found_first} and found_middle={found_middle} and found_last={found_last}")
                print()
                assert False
            continue
        text[0] = text[0][1:]
        if (found_he or found_she or found_they) and not full_name_not_found:
            for i in range(1, len(text)):
                if nname in text[i]:
                    if text[i].startswith(nname):
                        text[i] = text[i].replace(nname, 'She' if found_she else 'He' if found_he else 'They' if found_they else 'It', 1)
                    text[i] = text[i].replace(' '+nname+' ', ' she ' if found_she else ' he ' if found_he else ' they ' if found_they else ' it ')

        text = '. '.join(text) + '.'
        for x in spe_period:
            y = x.replace('.','#')
            text = text.replace(y,x)
        text = ' '+text # add a space before first name
        return text.strip()

    raise RuntimeError("augmentation_permutation2 failed without producing a valid permutation")