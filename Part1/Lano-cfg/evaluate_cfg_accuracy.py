import os
import torch
import random
import argparse
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_cfg import CFG_Config

def sample_sequence(model, prompt_ids, max_new_tokens=512, eos_token=0, temperature=1.0):
    """
    模型自回归推理基础实现。
    根据指定的生成策略(带温度的 multinomial 采样或者 greedy search)，自回归生成至遇到 eos_token 或者达到上限。
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            # 适配不同模型产出的格式差异
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            next_token_logits = logits[0, -1, :] 
            
            if temperature > 0:
                # 带温度缩放分布的采样，确保一定的生成多样性（如论文所述的多样性模式）
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy 最优解码
                next_token = torch.argmax(next_token_logits).item()
                
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            # 完整生成一个独立序列即停止
            if next_token == eos_token:
                break
                
    return input_ids[0].cpu().tolist()

def eval_accuracy(config, generated_seq, bos_token, eos_token):
    """
    判断模型生成的这串 token 能否正确满足原有 CFG 。
    逻辑：去除头尾(BOS/EOS), 交由快速验证版 DP 'solve_dp_noneq_fast' 做严格语法语义合法性鉴定。
    """
    if len(generated_seq) < 2 or generated_seq[-1] != eos_token:
        # 如果模型并未按预期吐出结尾符（被长度上限截断等），算作格式不符
        return False
        
    # 脱去左侧可能混含的 BOS 与 右侧唯一的 EOS 截取纯序列 
    # (如果 prompt 的左侧含有隐式的 BOS 已经被剥离处理)
    if generated_seq[0] == bos_token:
        pure_seq = generated_seq[1:-1]
    else:
        pure_seq = generated_seq[:-1]
        
    if len(pure_seq) == 0:
        return False
        
    # count, dp_sol, _, possibility
    count, *_ = config.solve_dp_noneq_fast(pure_seq, no_debug=True)
    
    # count == 0 代表序列零错误（100% 贴合 CFG 语法树分支要求）
    return count == 0

def main():
    parser = argparse.ArgumentParser(description="评估基于 CFG 训练出来的 GPT 模型准确率 (论文 Fig 4 复现脚本)")
    parser.add_argument("--config_path", type=str, default="configs/cfg3f.json")
    parser.add_argument("--test_data_path", type=str, default="datasets/test.parquet", help="加载的测试集数据文件路径")
    parser.add_argument("--model_path", type=str, default="/ruilab/jxhe/CoE_Monitor/checkpoints/GPT_2_Small", help="模型目录")
    parser.add_argument("--num_eval_samples", type=int, default=20000, help="论文为每个场景均生成超过 20,000 个样本评估")
    parser.add_argument("--prefix_len", type=int, default=50, help="给定的前缀补充长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    
    args = parser.parse_args()

    # 1. 挂载被测评的 CFG 环境以及特殊 Tokens
    config = CFG_Config.from_graph(args.config_path)
    
    import json
    model_config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        BOS_TOKEN = model_config.get("bos_token_id", 0)
        EOS_TOKEN = model_config.get("eos_token_id", 0)
    else:
        BOS_TOKEN = config.num_sym + 1
        EOS_TOKEN = 0

    # 1.5 读取构建好的测试集以替代之前现场生成
    import pandas as pd
    if not os.path.exists(args.test_data_path):
        print(f"找不到测试集文件 {args.test_data_path}，请先运行 build_parquet_dataset.py 生成。")
        return
        
    df_test = pd.read_parquet(args.test_data_path)
    test_samples = df_test['input_ids'].tolist()
    # 根据数据集长度调整评测总数
    num_eval = min(args.num_eval_samples, len(test_samples))
    print(f"已加载 {len(test_samples)} 条测试数据。将评测 {num_eval} 条样本。")

    # -------------------------------------------------------------
    # TODO: 请在这个位置引入并唤醒您刚才写出的或是已经预训练好的模型！
    # 示例:
    # from modeling_gpt2_variants import CustomGPT2Model
    # from transformers import GPT2Config
    # # 参考论文：12 heads, 12 layers, 768 dim, seq=512 -> ~86M
    # model_config = GPT2Config(vocab_size=config.num_sym+2, n_embd=768, n_layer=12, n_head=12, n_positions=512)
    # model = CustomGPT2Model(model_config, gpt_type='standard')
    # model.load_state_dict(torch.load("path/to/checkpoint.pt"))
    # model.cuda()
    # -------------------------------------------------------------
    
    model = None # 待替换实例
    if model is None:
        print("【拦截警告】：尚未实例化模型。此评测脚本需填充你预训练出来的模型对象进入内存中运行，请补充上方 TODO 注释代码区！")
        return

    rng = random.Random(2026)
    correct_unconditional = 0
    correct_conditional = 0
    
    # 按照原论文，我们在A100等集群使用无重叠高自由度测试覆盖评估 
    print(f"开始启动对 {num_eval} 个测试样本的准确率评估...")
    
    for i in tqdm(range(num_eval)):
        # ========================================================
        # 评测场景 1：完全无内容前置条件（仅喂入 BOS）
        # ========================================================
        gen_seq_uncond = sample_sequence(
            model=model, 
            prompt_ids=[BOS_TOKEN], 
            max_new_tokens=512, 
            eos_token=EOS_TOKEN,
            temperature=args.temperature
        )
        if eval_accuracy(config, gen_seq_uncond, BOS_TOKEN, EOS_TOKEN):
            correct_unconditional += 1
                
        # ========================================================
        # 评测场景 2：给定长度 50 的真实合法 CFG 序列前缀 (Prompt Completion)
        # ========================================================
        sample_ids = test_samples[i]
        
        # 剥离测试集序列中的 BOS 与 EOS
        gt_seq = sample_ids[:]
        if len(gt_seq) > 0 and gt_seq[0] == BOS_TOKEN:
            gt_seq = gt_seq[1:]
        if len(gt_seq) > 0 and gt_seq[-1] == EOS_TOKEN:
            gt_seq = gt_seq[:-1]
            
        # 截取前 args.prefix_len
        prefix = gt_seq[:args.prefix_len] if len(gt_seq) > args.prefix_len else gt_seq
            
        gen_seq_cond = sample_sequence(
            model=model,
            prompt_ids=[BOS_TOKEN] + prefix,
            max_new_tokens=512 - len(prefix),
            eos_token=EOS_TOKEN,
            temperature=args.temperature
        )
        if eval_accuracy(config, gen_seq_cond, BOS_TOKEN, EOS_TOKEN):
            correct_conditional += 1

    acc_uncond = correct_unconditional / num_eval
    acc_cond = correct_conditional / num_eval
    
    print("-" * 50)
    print("【评测最终报告】")
    print(f"场景一 (无条件, 即生成完整句子) Accuracy: {acc_uncond * 100:.4f}%")
    print(f"场景二 (有条件, 馈送前 {args.prefix_len} 个 Tokens 且符合语境) Accuracy: {acc_cond * 100:.4f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()