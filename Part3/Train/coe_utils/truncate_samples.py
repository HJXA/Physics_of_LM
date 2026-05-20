import os
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
torch.set_num_threads(1)          # 限制 intra-op
torch.set_num_interop_threads(1)  # 限制 inter-op（可选但推荐）

def process_file(file_path, keep_samples):
    try:
        # Load with weights_only=False as in the analysis script
        tensor = torch.load(file_path, map_location="cpu", weights_only=False)
        
        # tensor shape: (Batch, Layer, Dim)
        if tensor.shape[0] > keep_samples:
            new_tensor = tensor[:keep_samples].clone()
            torch.save(new_tensor, file_path)
            return "processed"
        else:
            return "skipped"
        
    except Exception as e:
        return f"failed: {str(e)}"

def main():
    # 路径与 one_model_ASI.py 中保持一致
    BASE_MODEL_DIR = "/ruilab/jxhe/CoE_Monitor/ms-swift/coe_train_result/PT_HJXA_Llama_104M_Minimind_no_packing_no_padding_free"
    
    if not os.path.exists(BASE_MODEL_DIR):
        print(f"Path does not exist: {BASE_MODEL_DIR}")
        return

    # 查找所有 run 目录
    run_dirs = [
        d for d in os.listdir(BASE_MODEL_DIR)
        if os.path.isdir(os.path.join(BASE_MODEL_DIR, d)) and d.startswith("v")
    ]

    if len(run_dirs) == 0:
        print("No run directory found")
        return

    # 选择最新的 run
    latest_run = sorted(run_dirs)[-1]
    DATA_DIR = os.path.join(BASE_MODEL_DIR, latest_run, "Layer_Hidden_Train")
    
    print(f"Target Directory: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory does not exist: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pt")]
    print(f"Found {len(files)} .pt files.")
    
    KEEP_SAMPLES = 10
    count_processed = 0
    count_skipped = 0
    count_failed = 0
    
    # 设定并发数，可以根据机器核心数调整
    max_workers = min(16, os.cpu_count() or 4)
    print(f"Using {max_workers} workers.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, os.path.join(DATA_DIR, f), KEEP_SAMPLES): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Truncating files"):
            fname = futures[future]
            try:
                result = future.result()
                if result == "processed":
                    count_processed += 1
                elif result == "skipped":
                    count_skipped += 1
                else:
                    count_failed += 1
                    print(f"Error processing {fname}: {result}")
            except Exception as e:
                count_failed += 1
                print(f"Exception in worker for {fname}: {e}")
            
    print(f"Done. Processed (truncated): {count_processed}, Skipped (already <= {KEEP_SAMPLES}): {count_skipped}, Failed: {count_failed}")


if __name__ == "__main__":
    main()
