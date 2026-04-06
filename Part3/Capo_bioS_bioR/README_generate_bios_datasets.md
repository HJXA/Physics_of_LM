# generate_bios_datasets.py 使用说明与运行逻辑

本文档详尽介绍 `generate_bios_datasets.py` 脚本的目的、运行流程、生成的数据集结构、示例输出与实际使用方法。

**文件位置**: [generate_bios_datasets.py](Part3/Capo_bioS_bioR/generate_bios_datasets.py)

## 一、脚本完成的任务（概述）

- 生成一个基于规则模板的传记类数据集（Capo 规则流程），主要产生人物基础记录与多种规则增强版本。
- 产出两种并行格式的数据：结构化字典（parquet）和纯文本（parquet），并在 `samples` 子目录中同步导出供人工抽查的 `.json`/.txt 文件。
- 基于基础人物信息再生成六类 QA 文本（字符串格式）。

总体目标是生成可复现、规则可追溯的训练/评估数据集（不依赖模型重写，仅使用规则函数 `get_text_simple3` 与 `augmentation_permutation2`）。

## 二、会生成哪些数据集（目录与格式）

默认输出根目录为脚本同级的 `datasets/`（可用 `--output_dir` 改写）。输出结构如下：

- `datasets/dict/<dataset_name>/part_*.parquet`：结构化字典版本（包含人物属性与 `text` 字段）。
- `datasets/text/<dataset_name>/part_*.parquet`：仅包含 `text` 字段的纯文本版本（训练输入）。
- `datasets/QA/<qa_name>/data.parquet`：QA 字符串数据集（text parquet）。
- `datasets/samples/...`：镜像目录下的检验样本，文本为 `.txt`，字典为 `.json`，每个 parquet 导出前 `--show_samples` 条用于人工核验。

主要生成的数据集名称：

- `bioS_single`：基础 6 句传记（每人一条），part_1（dict/text）
- `bioS_single_fullname`：将代词替换为全名的变体
- `bioS_single_permute`：规则置换（permutation）产生的多个 part（`--max_permute` 控制数量）
- `bioS_single_permute_fullname`：在 permute 基础上把代词替换为全名
- `bioS_multi`：对同一人重新采样模板生成的新文本（多 part，由 `--max_multi` 控制）
- `bioS_multi_fullname`：multi 的 fullname 版本
- `bioS_multi_permute`：multi 上的规则置换
- `bioS_multi_permute_fullname`：multi_permute 的 fullname 版本
- `QA/q1_birth_date` ... `QA/q6_company_city`：六类 QA 文本（以 `data.parquet` 存放）

每个 `*_permute` 与 `*_multi` 会按 part 索引分文件（例如 `part_1.parquet`、`part_2.parquet`）。

## 三、脚本运行流程（高层步骤）

1. 生成 `bioS_single` 基础记录：调用 `build_base_records` 生成 `num_samples` 个唯一全名的人物档案，并用 `get_text_simple3` 合成 6 句传记，保存为 dict/text 两种 parquet。
2. 从刚写入的 dict parquet 回读基础记录，确保后续所有增强基于落盘数据（可复现、可审计）。
3. 生成 `single` 与 `multi` 系列的所有规则增强：包括 fullname 版本、permute 版本和 permute+fullname，同时按 `part_idx` 划分输出文件。
4. 基于 base 信息生成 6 类 QA（文本形式）。

脚本运行时会在控制台输出 1/4..4/4 的进度信息并在完成后列出目录结构说明。

## 四、关键函数说明（源码内调用关系）

- `load_lines(filename)`：从 `fields/` 目录读取词表（每行一个项）。
- `load_companies()`：读取 `company.txt`（格式 `公司名; 城市`）并返回字典列表。
- `build_unique_people(num_samples, seed)`：基于各个词表随机采样生成唯一 `full_name` 的人物档案数组（含出生日期、大学、公司、代词等）。
- `get_text_simple3`：外部导入的规则模板合成函数，用来把人物属性渲染为 6 句传记。
  - 新增参数 `fixed_templates`（默认 `False`）。当 `fixed_templates=True` 时，函数会固定使用每个模板池的第一个句式，
    用于 `bioS_single` 基础数据生成以保证每条样本的结构一致；当为 `False`（如 `multi` 阶段），仍采用随机采样模板以增加多样性。
- `render_single_bio(person)`：封装 `get_text_simple3` 并确保代词/性别一致性，返回 6 句传记字符串。
- `augmentation_permutation2(person, source_text)`：外部导入的规则置换函数，用于生成 permute 变体（不调用模型）。
- `permute_text(person, source_text, seed)`：为每次置换设置独立随机种子并调用 `augmentation_permutation2`，保证可复现。
- `replace_pronouns_with_fullname(text, full_name)`：把句子里的代词替换为 `full_name`（fullname 版本）。
- `save_dual_dataset(records, datasets_root, samples_root, dataset_name, part_idx, n_samples)`：同时写入 dict 与 text parquet，并导出 samples（json/txt）。

## 五、示例输出（典型记录）

1) 结构化字典记录（parquet 中一条记录示例，JSON 格式）：

```json
{
  "id": 123,
  "first_name": "Alice",
  "middle_name": "B.",
  "last_name": "Chen",
  "full_name": "Alice B. Chen",
  "birthmonth": "March",
  "birthday": "12",
  "birthyear": "1990",
  "birthcity": "Shanghai",
  "university": "Fudan University",
  "field": "Computer Science",
  "company1name": "Acme Corp",
  "company1city": "Beijing",
  "pronoun": "She",
  "dataset": "bioS_single",
  "variant_index": 1,
  "text": "She graduated from Fudan University with a degree in Computer Science. She joined Acme Corp in Beijing..."
}
```

2) 文本样本（`datasets/text/.../part_1.parquet` 导出的 `.txt` 示例行）：

```
She graduated from Fudan University with a degree in Computer Science. She worked at Acme Corp in Beijing. ...
```

3) QA 示例（`datasets/QA/q1_birth_date/data.parquet` 中的一条文本）：

```
What is the birth date of Alice B. Chen? Answer: March 12, 1990.
```

## 六、运行与使用示例

基本运行命令（在脚本目录下执行）：

```bash
python generate_bios_datasets.py --num_samples 100000 --output_dir ./datasets --seed 42 --show_samples 10 --max_permute 5 --max_multi 5
```

快速测试（生成 10 条并输出样本用于人工检查）：

```bash
# 在脚本目录下执行，生成 10 条基础样本并把抽查样本导出到 datasets_test/samples
python generate_bios_datasets.py --num_samples 10 --output_dir ./datasets_test --seed 42 --show_samples 10 --max_permute 3 --max_multi 2
```

说明：基于代码修改，`bioS_single` 阶段会以 `fixed_templates=True` 固定每个模板池的首项（即每条样本的句式结构一致），
而 `bioS_multi` / 后续增强仍然采用随机模板以保证多样性。

常见用法：

- 只想快速跑小样本测试：把 `--num_samples` 调小（例如 1000 或 10000）。
- 控制每个 parquet 导出的抽查样本条数：`--show_samples`（默认 10）。
- 控制 permute / multi 的 part 数量：`--max_permute` 与 `--max_multi`。

在 Python 中读取 parquet 示例：

```python
import pandas as pd
df = pd.read_parquet('datasets/dict/bioS_single/part_1.parquet')
print(df.columns)
print(df.loc[0].to_dict())
```

注意：读取 parquet 需要安装 `pyarrow` 或 `fastparquet`。推荐安装：

```bash
pip install pandas pyarrow
```

## 七、前置条件与注意事项

- 依赖的词表文件必须存在于 `fields/` 目录（脚本同级）。例如 `first_name.txt`、`middle_name.txt`、`last_name.txt`、`university.txt`、`field.txt`、`city.txt`、`company.txt` 等。
- 依赖模块 `Capo_bioS_bioR` 中应提供 `get_text_simple3` 与 `augmentation_permutation2` 两个函数（脚本注释已说明仅依赖这两个核心函数）。
- 安装依赖：`pandas`、`numpy`、`pyarrow`。建议 Python 3.8+。
- 随机种子控制：脚本会在不同阶段（base、permute、multi）使用不同的偏移种子以保证可复现且避免相互影响，若需完全相同输出请确保 `--seed` 一致。

## 八、常见问题排查

- 如果 `pd.read_parquet` 报错找不到 engine，请安装 `pyarrow`：`pip install pyarrow`。
- 如果报错找不到词表文件，检查 `fields/` 目录是否存在且文件名拼写正确。
- 如果 `augmentation_permutation2` 返回空导致异常，检查该函数实现及输入文本是否符合预期（脚本会抛 RuntimeError）。

## 九、建议的后续步骤

- 若需要模型重写增强（非规则），建议把模型增强作为可选分支并保留当前规则流程以便对照评估。
- 可在生成后运行样本质量检测脚本（例如检测代词一致性、语法错误、信息泄漏等）。

---

若需我把 README 另存为不同文件名，或把其中某段扩展为单独的开发文档（如字段说明详表、QA 分布统计脚本），我可以继续添加。
