# export PATH="/ruilab/jxhe/miniconda3/envs/PoL/bin:$PATH"
import os
import random
import argparse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import multiprocessing as mp

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_cfg import CFG_Config


def _make_fixed_length_sample(seq, bos_token_id, eos_token_id, pad_token_id, chunk_size):
    sample = [bos_token_id] + seq + [eos_token_id]
    if len(sample) >= chunk_size:
        sample = sample[:chunk_size]
        sample[-1] = eos_token_id
        return sample
    sample.extend([pad_token_id] * (chunk_size - len(sample)))
    return sample

def _worker_generate_chunks(args):
    config_path, worker_id, seed, num_chunks, bos_token_id, eos_token_id, pad_token_id, save_dir, chunk_size, cache_size = args
    if num_chunks <= 0:
        return 0
        
    config = CFG_Config.from_graph(config_path)
    rng = random.Random(seed)
    
    EOS_TOKEN = eos_token_id
    BOS_TOKEN = bos_token_id
    PAD_TOKEN = pad_token_id

    chunks = []
    total_generated = 0
    save_path = os.path.join(save_dir, f"part-train-{worker_id:04d}.parquet")
    writer = None
    
    # 动态内部种子计数器
    sample_count = 0
    
    def flush_to_disk(chunks_to_write):
        nonlocal writer
        df = pd.DataFrame({'input_ids': chunks_to_write})
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(save_path, table.schema)
        writer.write_table(table)
            
    while total_generated < num_chunks:
        current_rng = random.Random(seed + sample_count)
        seq = config.generate_onedata_pure(current_rng)
        sample_count += 1
        chunk = _make_fixed_length_sample(seq, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, chunk_size)
        chunks.append(chunk)
        total_generated += 1

        if total_generated % 10000 == 0:
            print(f"[Worker {worker_id}] 已生成: {total_generated} / {num_chunks} 条训练样本")

        if len(chunks) >= cache_size:
            flush_to_disk(chunks)
            chunks.clear()
                
    if len(chunks) > 0:
        flush_to_disk(chunks)
        chunks.clear()
        
    if writer is not None:
        writer.close()
    return total_generated

def generate_continuous_chunks_to_parquet(config_path, seed, num_chunks, bos_token_id, eos_token_id, pad_token_id, save_dir, chunk_size=512, cache_size=100000, num_workers=8):
    print(f"启动 {num_workers} 个进程并行生成总计 {num_chunks} 条训练样本（每条固定长度 {chunk_size}）...")
    pool = mp.Pool(num_workers)
    
    base_chunks = num_chunks // num_workers
    remainder = num_chunks % num_workers
    
    seed_step = base_chunks + 10000
    
    tasks = []
    for i in range(num_workers):
        n_chunks = base_chunks + (1 if i < remainder else 0)
        worker_seed = seed + i * seed_step
        tasks.append((config_path, i, worker_seed, n_chunks, bos_token_id, eos_token_id, pad_token_id, save_dir, chunk_size, cache_size))
        
    results = pool.map(_worker_generate_chunks, tasks)
    pool.close()
    pool.join()
    return sum(results)

def _worker_generate_samples(args):
    config_path, worker_id, seed, num_samples, bos_token_id, eos_token_id, save_dir, cache_size = args
    if num_samples <= 0:
        return 0
        
    config = CFG_Config.from_graph(config_path)
    samples = []
    save_path = os.path.join(save_dir, f"part-test-{worker_id:04d}.parquet")
    writer = None
    
    def flush_to_disk(samples_to_write):
        nonlocal writer
        df = pd.DataFrame({'input_ids': samples_to_write})
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(save_path, table.schema)
        writer.write_table(table)
            
    for i in range(num_samples):
        current_rng = random.Random(seed + i)
        seq = config.generate_onedata_pure(current_rng)
        sample = [bos_token_id] + seq + [eos_token_id]
        samples.append(sample)
        
        if (i + 1) % 5000 == 0:
            print(f"[Worker {worker_id}] 已生成: {i + 1} / {num_samples} 条测试样本")
            
        if len(samples) >= cache_size:
            flush_to_disk(samples)
            samples.clear()

    if len(samples) > 0:
        flush_to_disk(samples)
        samples.clear()
        
    if writer is not None:
        writer.close()
    return num_samples

def generate_independent_samples_to_parquet(config_path, seed, num_samples, bos_token_id, eos_token_id, save_dir, cache_size=100000, num_workers=8):
    print(f"启动 {num_workers} 个进程并行生成总计 {num_samples} 条独立测试样本（不做 padding）...")
    pool = mp.Pool(num_workers)
    
    base_samples = num_samples // num_workers
    remainder = num_samples % num_workers
    
    # 动态计算安全跨度：单进程直接生成独立样本的数量 + 安全冗余
    seed_step = base_samples + 10000 
    
    tasks = []
    for i in range(num_workers):
        n_samples = base_samples + (1 if i < remainder else 0)
        worker_seed = seed + i * seed_step
        tasks.append((config_path, i, worker_seed, n_samples, bos_token_id, eos_token_id, save_dir, cache_size))
        
    results = pool.map(_worker_generate_samples, tasks)
    pool.close()
    pool.join()
    return sum(results)

def save_sample_txt(data, config, bos_token_id, eos_token_id, save_path, num_samples=10):
    """
    保存起始的若干条样本为纯文本文件，以便肉眼检查数据格式。
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Vocab Config - Number of symbols (T): {config.num_sym}\n")
        f.write(f"In this run: BOS is {bos_token_id}, EOS is {eos_token_id}\n")
        f.write("-" * 50 + "\n")
        for i in range(min(num_samples, len(data))):
            f.write(f"Sample {i+1} (Length: {len(data[i])}):\n")
            f.write(str(data[i]) + "\n\n")

def merge_parquet_files(src_dir, dest_file):
    print(f"正在合并 {src_dir} 下的分区文件到单一文件 {dest_file} ...")
    dataset = pq.ParquetDataset(src_dir)
    table = dataset.read()
    pq.write_table(table, dest_file)
    # 合并完成后清理临时目录及其子文件
    import shutil
    shutil.rmtree(src_dir)
    print(f"合并完成且已清理临时文件夹: {src_dir}")

def main():
        
    import shutil
    parser = argparse.ArgumentParser(description="生成 CFG Dataset 并保存为 Parquet 格式")
    parser.add_argument("--config_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/data-synthetic-pretrain/Lano-cfg/configs/cfg3b.json", help="使用的 CFG 规则配置文件")
    parser.add_argument("--save_dir", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/datasets/512_Padding/cfg3b", help="数据的输出保护路径")
    parser.add_argument("--model_path", type=str, default="/ruilab/jxhe/CoE_Monitor/checkpoints/GPT_2_Small", help="模型路径，用于获取bos和eos")
    
    # 论文中无限数据范式相当于使用全量随机构造。1亿 token 大约对应约 195,000 个长度为 512 的 sequence chunks。
    # 您可按需扩展为 49亿 tokens 对应的预训练上限 (~9,500,000 chunks)
    parser.add_argument("--train_chunks", type=int, default=20000000, help="训练样本总条数 (每条固定长度为 chunk_size)")
    parser.add_argument("--test_samples", type=int, default=40000, help="测试集独立序列样本数") 
    parser.add_argument("--chunk_size", type=int, default=512, help="固定窗口长度，论文中设为 512") # 
    parser.add_argument("--num_workers", type=int, default=16, help="并行生成的进程数量")
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    

    bos_token_id = 0
    eos_token_id = 4
    pad_token_id = 5
    print(f"bos_token_id: {bos_token_id}, eos_token_id: {eos_token_id}")

    # 1. 加载 CFG 规则
    config = CFG_Config.from_graph(args.config_path)
    print("CFG 载入完成. Number of symbols (Terminals):", config.num_sym)
    
    # 2. 生成 Train 训练集
    print("\n--- 构建训练集 ---")
    train_out_final = os.path.join(args.save_dir, "train.parquet")
    train_out_tmp = os.path.join(args.save_dir, "train_tmp_parquet")

    if os.path.exists(train_out_tmp):
        shutil.rmtree(train_out_tmp)
    os.makedirs(train_out_tmp, exist_ok=True)
            
    generate_continuous_chunks_to_parquet(
        config_path=args.config_path, 
        seed=42, 
        num_chunks=args.train_chunks, 
        bos_token_id=bos_token_id, eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        chunk_size=args.chunk_size,
        save_dir=train_out_tmp,
        cache_size=100000,
        num_workers=args.num_workers
    )
    
    # 统一合并成唯一 Parquet
    merge_parquet_files(train_out_tmp, train_out_final)
    print(f"预训练集已妥善保存至单一文件: {train_out_final}")
    
    # 额外保存前 10 条测试用的明文 (读取构建好数据集的开头展示即可)
    try:
        df_train_head = pd.read_parquet(train_out_final, engine='pyarrow').head(10)
        train_sample_txt = os.path.join(args.save_dir, "train_samples_head10.txt")
        save_sample_txt(df_train_head['input_ids'].tolist(), config, bos_token_id, eos_token_id, train_sample_txt, num_samples=10)
        print(f"前 10 条训练样本明文已导出至: {train_sample_txt}")
    except Exception as e:
         print(f"提取前10条明文失败: {e}")
    
    # 3. 生成 Test 测试集
    print("\n--- 构建测试集 ---")
    test_out_final = os.path.join(args.save_dir, "test.parquet")
    test_out_tmp = os.path.join(args.save_dir, "test_tmp_parquet")
    
    if os.path.exists(test_out_tmp):
        shutil.rmtree(test_out_tmp)
    os.makedirs(test_out_tmp, exist_ok=True)
            
    # 测试集不切分 chunk，生成直接可用的独立样本
    generate_independent_samples_to_parquet(
        config_path=args.config_path, 
        seed=1_000_000_042, 
        num_samples=args.test_samples, 
        bos_token_id=bos_token_id, eos_token_id=eos_token_id,
        save_dir=test_out_tmp,
        cache_size=100000,
        num_workers=args.num_workers
    )
    
    merge_parquet_files(test_out_tmp, test_out_final)
    print(f"测试数据已保存至单一文件: {test_out_final}")
    
    # 额外保存前 10 条测试数据到 text 中用于检查
    try:
        df_test_head = pd.read_parquet(test_out_final, engine='pyarrow').head(10)
        test_sample_txt = os.path.join(args.save_dir, "test_samples_head10.txt")
        save_sample_txt(df_test_head['input_ids'].tolist(), config, bos_token_id, eos_token_id, test_sample_txt, num_samples=10)
        print(f"前 10 条测试样本明文已导出至: {test_sample_txt}")
    except Exception as e:
         print(f"提取前10条明文失败: {e}")

if __name__ == "__main__":
    main()
