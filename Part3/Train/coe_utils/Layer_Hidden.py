import torch
import os


def Layer_Hidden_Train(hidden_states, eos_token_id=None,pad_token_id=None, labels=None, loss_label=False, steps=1, rank=0, input_ids=None):
    """
    处理模型训练时的隐藏状态。
    两种模式：
    1. Packed Mode (Padding-free): 如果 Batch=1 且 input_ids 含有 eos_token_id，
       则根据 eos 分割样本，每个样本计算全量 Token 的平均 Hidden State。
    2. Normal Mode: 常规 Batch/Seq 处理，根据 labels != -100 或全量作为掩码计算平均。
    
    参数:
        hidden_states: Tuple/List of Tensors (Layer, Batch, Seq, Dim)
        labels: Tensor (Batch, Seq)
        input_ids: Tensor (Batch, Seq) 必传 - 用于 packed 模式对齐与切分
    返回:
        hidden_states_np: Tensor, 形状 (Batch_Real, Layer, Hidden_Dim)
    """

    if input_ids is None:
        raise ValueError("input_ids 不能为空！当前逻辑要求始终传入 input_ids。")


    layer_0_states = hidden_states[0]
    if steps == 1 and rank == 0:
        print(f"hidden_states.len == Layer_Num == ", len(hidden_states))
        if labels is not None:
            print(f"labels 中 -100 比例", (labels == -100).float().mean())
    batch_size, seq_len, hidden_dim = layer_0_states.shape
    device = layer_0_states.device
    dtype = layer_0_states.dtype 
    num_layers = len(hidden_states)
    del layer_0_states  # 释放内存

    # 仅基于 input_ids 判定 Packed Mode（简化逻辑）
    flat_inputs = input_ids.view(-1)
    has_eos = bool(eos_token_id is not None)
    is_packed = (batch_size == 1 and labels.shape != (batch_size, seq_len) and has_eos)

    if steps == 1 and rank == 0:
        if is_packed:
            print(f"检测到 Packed 模式: Batch=1 且label形状不匹配 且 Input 包含 EOS")
        else:
            print(f"Packed 模式条件不满足: Batch={batch_size}, SeqLen={seq_len}, Labels_Shape={labels.shape if labels is not None else None}, EOS_Token_ID={eos_token_id}")

    if is_packed:
        # === Packed Mode (Padding-free, optimized) ===

        eos_positions = (flat_inputs == eos_token_id).nonzero().view(-1) 
        ends = eos_positions + 1
        if ends.numel() == 0 or ends[-1].item() != flat_inputs.numel(): # 如果没有 EOS 或最后一个 EOS 不是序列末尾，添加一个虚拟的结束位置
            ends = torch.cat([ends, torch.tensor([flat_inputs.numel()], device=device, dtype=ends.dtype)])

        starts = torch.cat([torch.zeros(1, device=device, dtype=ends.dtype), ends[:-1]])
        real_batch = ends.numel()
        

        if steps == 1 and rank == 0:
           
            total_hidden_used = int((ends - starts).sum().item())
            print(f"[Packed 模式] 已触发。通过 EOS={eos_token_id} 共恢复样本数: {real_batch}")
            
            # --- 验证 1: hidden_intervals 覆盖检查 ---
            check_ptr = 0
            has_gap = False
            for idx in range(real_batch): # 402
                h_s = int(starts[idx].item())
                h_e = int(ends[idx].item())
                if h_s != check_ptr:
                     print(f"  [Error] 样本 {idx} 前存在空缺! {check_ptr} -> {h_s}")
                     has_gap = True
                check_ptr = h_e
            
            if not has_gap:
                print(f"  [验证 1] hidden_intervals 覆盖连续性检查通过 (End={check_ptr}, SeqLen={seq_len})")
            else:
                 print(f"  [验证 1] 存在空缺！(End={check_ptr}, SeqLen={seq_len})")

            # --- 验证 2: 总长度映射 ---
            print(f"[Packed 模式] 长度映射: EOS切分后总长={total_hidden_used} (Hidden SeqLen={seq_len})")
            if total_hidden_used != seq_len:
                print(f"[警告][验证二] 映射后的 Hidden 总长度 ({total_hidden_used}) 与 实际 SeqLen ({seq_len}) 不一致!")

            print("-" * 50)

        starts_list = starts.cpu().tolist()
        # ends_list 不需要冗余的 int 转换（tolist() 已返回 Python int）
        ends_list = [min(e, seq_len) for e in ends.cpu().tolist()]

            
        final_tensor = torch.zeros(num_layers, real_batch, hidden_dim, dtype=dtype, device=device)
        
        # 使用 starts/ends 直接切分
        for i, layer_state in enumerate(hidden_states):
            # 处理 detach 和 dtype 转换（仅当需要时）
            if not loss_label:
                layer_state = layer_state.detach()
            if layer_state.dtype != dtype:
                layer_state = layer_state.to(dtype)

            flat_state = layer_state.squeeze(0)  # (seq_len, hidden_dim)

            for j, (h_start, h_end) in enumerate(zip(starts_list, ends_list)):
                if h_end > h_start:
                    segment = flat_state[h_start:h_end]          # (seg_len, hidden_dim)
                    final_tensor[i, j] = segment.mean(dim=0)            # 直接赋值

            del flat_state  # 释放内存
        if steps == 1 and rank == 0:
            print(f"final_tensor 形状 (Layer, Batch, Dim): {final_tensor.shape} (Real_Batch={real_batch})")

        return final_tensor.permute(1, 0, 2)


    else:
        # === Normal Mode === # no packing / SFT
        # 预先创建存储
        all_layers_mean = torch.zeros(
            (num_layers, batch_size, hidden_dim), 
            device=device, 
            dtype=dtype
        )

        # 创建掩码
        # 修改逻辑：仅 Mask 掉 Padding 部分的 -100，保留 Prompt 中间的 -100。
        # 假设 Padding 总是位于 Sequence 的末尾。
        # 我们可以找到每个样本中最后一个非 -100 的位置，该位置之前的所有 Token (包括 -100) 都视为有效，之后视为 Padding。
        
        if labels is not None and labels.shape == (batch_size, seq_len):
            if steps == 1 and rank == 0: 
                print(f"[普通模式] 使用 labels 末尾 Padding 作为掩码 (保留 Prompt 中的 -100[SFT也是这样])。")
                print(" 找到最后一个非 -100 的位置索引")
            
            # (B, S)
            is_valid = (labels != -100)
            
            # 找到最后一个非 -100 的位置索引
            # 方法：构造 range，mask 掉无效位，取 max
            # 注意：如果全为 -100，则长度为 0
            
            seq_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) # (B, S)
            
            # 将无效位置的索引设为 -1，有效位置保持索引值
            valid_indices = torch.where(is_valid, seq_indices, -1)
            
            # 对每个 batch 取最大索引，即为有效序列的结束位置 (包含)
            last_valid_index = valid_indices.max(dim=1).values # (B,) # 每个样本找到最后一个非 -100 的位置索引
            
            # 生成 Mask：索引 <= last_valid_index 的位置为 1
            mask = (seq_indices <= last_valid_index.unsqueeze(1)).float()
            
        elif pad_token_id is not None and input_ids is not None:
            # 使用 pad_token_id 和 input_ids 构造掩码
            if steps == 1 and rank == 0: 
                print(f"[普通模式] 使用 input_ids 中 pad_token_id={pad_token_id} 作为掩码。")
            is_pad = (input_ids == pad_token_id)
            mask = ~is_pad
        else:
            if steps == 1 and rank == 0: 
                print(f"[普通模式] Labels形状不匹配,且无法通过input_ids获取pad_token_id的掩码,使用全量掩码。")
            mask = torch.ones((batch_size, seq_len), device=device)

        valid_token_counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        mask_expanded = mask.unsqueeze(-1) # (B, S, 1)

        # 逐层处理
        for i, layer_state in enumerate(hidden_states):
            if not loss_label:
                layer_state = layer_state.detach()
            if layer_state.dtype != dtype:
                layer_state = layer_state.to(dtype)

            # (B, S, D) * (B, S, 1) -> Sum -> (B, D)
            masked_sum = (layer_state * mask_expanded).sum(dim=1)
            layer_mean = masked_sum / valid_token_counts
            all_layers_mean[i] = layer_mean

        return all_layers_mean.permute(1, 0, 2)



























if __name__ == "__main__":
    def test_layer_hidden_modes():
        print("\n=== Testing Layer_Hidden_Modes Consistency (Dynamic) ===")
        # 设置随机种子
        torch.manual_seed(42)
        
        # === 参数配置 ===
        B_real, D = 12, 18
        Layers = 10
        EOS_ID = 100
        PAD_ID = 0
        MIN_LEN = 7   # 每个样本最小长度
        MAX_LEN = 15 # 每个样本最大长度
        
        print(f"Config: Batch={B_real}, Dim={D}, Layers={Layers}, EOS={EOS_ID}")
        
        # =========== 步骤 1: 随机生成 B_real 个样本数据 ===========
        # 我们可以存储为 List of (Seq_Len, Input_Ids, Hidden_Dict_by_Layer)
        
        samples = []
        
        for i in range(B_real):
            # 随机长度 (不包含 EOS，因为稍后会追加)
            # 这里的长度指有效主要内容长度
            cur_len = torch.randint(MIN_LEN, MAX_LEN, (1,)).item()
            
            # 构造 Input IDs: [tok1, tok2, ..., EOS]
            # 实际有效长度变为 cur_len + 1 (因为 EOS 也是有效的一个 Token)
            # 为了简化，最后总是加上 EOS
            content_ids = torch.randint(1, 99, (cur_len,))
            sample_ids = torch.cat([content_ids, torch.tensor([EOS_ID])]) 
            real_seq_len = sample_ids.shape[0]
            
            # 为每一层生成随机 Hidden States (Seq, Dim)
            sample_hidden = {}
            for l in range(Layers):
                sample_hidden[l] = torch.randn(real_seq_len, D)
            
            samples.append({
                "ids": sample_ids,
                "hidden": sample_hidden,
                "len": real_seq_len
            })
            
        print(f"生成的样本长度: {[s['len'] for s in samples]}")
        
        # =========== 步骤 2: 构造 Normal Mode (Padding) 输入 ===========
        # 找出最大长度，用于 Padding
        max_seq_len = max([s["len"] for s in samples])
        
        norm_input_ids_list = []
        norm_labels_list = []
        norm_hidden_list_by_layer = [[] for _ in range(Layers)]
        
        # 预期结果 (手工计算)
        expected_means = torch.zeros(B_real, Layers, D)
        
        for idx, s in enumerate(samples):
            # 1. Input IDs Padding
            pad_len = max_seq_len - s["len"]
            pad_ids = torch.full((pad_len,), PAD_ID, dtype=torch.long)
            padded_input = torch.cat([s["ids"], pad_ids]) # [Max_S]
            norm_input_ids_list.append(padded_input)
            
            # 2. Labels Padding (-100MASK)
            # 有效部分为 1 (任意值 != -100)，无效部分为 -100
            # 注意: EOS 也是有效的一部分
            valid_labels = torch.ones(s["len"], dtype=torch.long)
            pad_labels = torch.full((pad_len,), -100, dtype=torch.long)
            padded_labels = torch.cat([valid_labels, pad_labels])
            norm_labels_list.append(padded_labels)
            
            # 3. Hidden States Padding
            # Padding 部分的 Hidden State 填 0 (或者随机数，反正应该被 Mask 掉)
            pad_hidden = torch.zeros(pad_len, D)
            
            for l in range(Layers):
                real_h = s["hidden"][l]
                padded_h = torch.cat([real_h, pad_hidden], dim=0) # [Max_S, D]
                norm_hidden_list_by_layer[l].append(padded_h)
                
                # 计算预期均值 (Mean over seq dim for this sample & layer)
                expected_means[idx, l, :] = real_h.mean(dim=0)
        
        # Stack to Tensor
        norm_input_ids = torch.stack(norm_input_ids_list) # (B, Max_S)
        norm_labels = torch.stack(norm_labels_list)       # (B, Max_S)
        
        # Tuple of Tensors (Layer, B, Max_S, D) - 实际上是 (B, M_S, D) per layer in tuple
        norm_hidden = tuple([torch.stack(batch_h_list) for batch_h_list in norm_hidden_list_by_layer])
        
        print(f"Normal Mode Input: Ids={norm_input_ids.shape}, "
              f"Hidden Layer0={norm_hidden[0].shape}")
        
        # =========== 步骤 3: 构造 Packed Mode (Concatenation) 输入 ===========
        
        # Concatenate Input IDs
        pack_input_ids = torch.cat([s["ids"] for s in samples]).unsqueeze(0) # (1, Total_S)
        
        # Concatenate Hidden States
        pack_hidden_list = []
        for l in range(Layers):
            concat_h = torch.cat([s["hidden"][l] for s in samples], dim=0) # (Total_S, D)
            pack_hidden_list.append(concat_h.unsqueeze(0)) # (1, Total_S, D)
            
        pack_hidden = tuple(pack_hidden_list)
        
        # Labels for trigger check
        # Label shape != (1, Total_S) to pass `batch_size == 1 and labels.shape != (batch_size, seq_len)` check
        # We can just pass a dummy 1D tensor
        pack_labels = torch.zeros(pack_input_ids.shape[1] + 1) # Just wrong shape
        
        print(f"Packed Mode Input: Ids={pack_input_ids.shape}, "
              f"Hidden Layer0={pack_hidden[0].shape}")
        
        # =========== 步骤 4: 执行并对比 ===========
        
        print("\n--- Running Normal Mode ---")
        out_normal = Layer_Hidden_Train(
            norm_hidden,
            eos_token_id=EOS_ID,
            labels=norm_labels,
            loss_label=False, # 测试 detach 逻辑
            input_ids=norm_input_ids,
            steps=1, rank=0
        )
        # out_normal shape: (B, L, D)
        
        print("\n--- Running Packed Mode ---")
        out_packed = Layer_Hidden_Train(
            pack_hidden,
            eos_token_id=EOS_ID,
            labels=pack_labels,
            loss_label=False,
            input_ids=pack_input_ids,
            steps=1, rank=0
        )
        # out_packed shape: (B, L, D)
        
        # =========== 5. 验证差异 ===========
        print("\n--- Comparison Results ---")
        
        # 1. 对比 Normal vs Packed
        diff_modes = (out_normal - out_packed).abs().max().item()
        print(f"Max Difference between Normal & Packed: {diff_modes:.8f}")
        
        # 2. 对比 Normal vs Expected (Manual)
        diff_expected = (out_normal - expected_means).abs().max().item()
        print(f"Max Difference between Normal & Manual Calc: {diff_expected:.8f}")
        
        if diff_modes < 1e-5:
            print(f"\n✅ 测试通过! (B={B_real}, L={Layers}, D={D})")
        else:
            print(f"\n❌ 测试失败! 差异过大: {diff_modes}")
            # 打印一些详细信息辅助调试
            # 找到差异最大的位置
            diff_tensor = (out_normal - out_packed).abs()
            max_val, max_idx = diff_tensor.view(-1).max(0)
            # idx -> (b, l, d)
            # flat index reconstruction
            # idx = b * (L*D) + l * D + d
            total_elements = B_real * Layers * D
            
            print(f"Debug Info: Normal Shape {out_normal.shape}, Packed Shape {out_packed.shape}")
            # 简单抽样几个样本对比
            for b_idx in range(min(3, B_real)):
                print(f"Sample {b_idx} Layer 0 Mean (First 3 dims):")
                print(f"  Norm:   {out_normal[b_idx, 0, :3].tolist()}")
                print(f"  Packed: {out_packed[b_idx, 0, :3].tolist()}")

    test_layer_hidden_modes()


