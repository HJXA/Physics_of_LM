import os
import random
import argparse
import pandas as pd
import pyarrow.parquet as pq

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_cfg import CFG_Config

def generate_continuous_chunks(config, seed, num_chunks, chunk_size=512):
    """
    核心思路 (依据论文)：
    对每个样本 x ~ L(G)，我们在其左侧拼接 BOS token，右侧拼接 EOS token。
    随后，我们将连续样本拼接，并随机切分数据，形成固定窗口长度为 512 的序列。
    """
    rng = random.Random(seed)
    
    # 按照作者提供的 DP 和序列惯例：
    # Terminals (T) 的 ID 通常是 1 到 num_sym。
    # 0 可以用作 EOS_TOKEN, 而 BOS_TOKEN 我们指派为 num_sym + 1
    EOS_TOKEN = 0
    BOS_TOKEN = config.num_sym + 1

    buffer = []
    chunks = []
    
    print(f"开始生成 {num_chunks} 个 sequence chunks (每个长度 {chunk_size})...")
    
    while len(chunks) < num_chunks:
        # 生成一条完全满足 CFG 的纯数据序列 (只包含 T)
        seq = config.generate_onedata_pure(rng)
        
        # 拼接 BOS 和 EOS
        buffer.append(BOS_TOKEN)
        buffer.extend(seq)
        buffer.append(EOS_TOKEN)
        
        # 按规定长度切分
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            chunks.append(chunk)
            buffer = buffer[chunk_size:]
            
            if len(chunks) % 10000 == 0:
                print(f"已生成: {len(chunks)} / {num_chunks} 块 (Chunks)")
            if len(chunks) >= num_chunks:
                break
                
    return chunks

def save_sample_txt(data, config, save_path, num_samples=10):
    """
    保存起始的若干条样本为纯文本文件，以便肉眼检查数据格式。
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Vocab Config - Number of symbols (T): {config.num_sym}\n")
        f.write(f"In this run: BOS is {config.num_sym + 1}, EOS is 0\n")
        f.write("-" * 50 + "\n")
        for i in range(min(num_samples, len(data))):
            f.write(f"Sample {i+1} (Length: {len(data[i])}):\n")
            f.write(str(data[i]) + "\n\n")

def main():
    parser = argparse.ArgumentParser(description="生成 CFG Dataset 并保存为 Parquet 格式")
    parser.add_argument("--config_path", type=str, default="configs/cfg3f.json", help="使用的 CFG 规则配置文件")
    parser.add_argument("--save_dir", type=str, default="datasets", help="数据的输出保护路径")
    
    # 论文中无限数据范式相当于使用全量随机构造。1亿 token 大约对应约 195,000 个长度为 512 的 sequence chunks。
    # 您可按需扩展为 49亿 tokens 对应的预训练上限 (~9,500,000 chunks)
    parser.add_argument("--train_chunks", type=int, default=200000, help="训练集序列总块数 (窗口大小为 512)")
    parser.add_argument("--test_chunks", type=int, default=5000, help="测试集序列块数")
    parser.add_argument("--chunk_size", type=int, default=512, help="固定窗口长度，论文中设为 512")
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. 加载 CFG 规则
    config = CFG_Config.from_graph(args.config_path)
    print("CFG 载入完成. Number of symbols (Terminals):", config.num_sym)
    
    # 2. 生成 Train 训练集
    print("\n--- 构建训练集 ---")
    train_data = generate_continuous_chunks(config, seed=42, num_chunks=args.train_chunks, chunk_size=args.chunk_size)
    df_train = pd.DataFrame({'input_ids': train_data})
    train_out = os.path.join(args.save_dir, "train.parquet")
    df_train.to_parquet(train_out, engine='pyarrow')
    print(f"预训练集已妥善保存至: {train_out}")
    
    # 额外保存前 10 条到同目录下的 text 文件中用于检查
    train_sample_txt = os.path.join(args.save_dir, "train_samples_head10.txt")
    save_sample_txt(train_data, config, train_sample_txt, num_samples=10)
    print(f"前 10 条训练样本明文已导出至: {train_sample_txt}")
    
    # 3. 生成 Test 测试集
    print("\n--- 构建测试集 ---")
    test_data = generate_continuous_chunks(config, seed=10042, num_chunks=args.test_chunks, chunk_size=args.chunk_size)
    df_test = pd.DataFrame({'input_ids': test_data})
    test_out = os.path.join(args.save_dir, "test.parquet")
    df_test.to_parquet(test_out, engine='pyarrow')
    print(f"测试数据已保存至: {test_out}")
    
    # 额外保存前 10 条测试数据到 text 中用于检查
    test_sample_txt = os.path.join(args.save_dir, "test_samples_head10.txt")
    save_sample_txt(test_data, config, test_sample_txt, num_samples=10)
    print(f"前 10 条测试样本明文已导出至: {test_sample_txt}")

if __name__ == "__main__":
    main()
