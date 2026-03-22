import math
import torch
from torch import nn
from modeling_gpt2 import GPT2Model, GPT2Attention, GPT2Block, GPT2Config

# ---------------------------------------------------------
# GPT_rot (GPT with Rotary Positional Embeddings - RoPE)
# 对应论文中的 GPT_rot 架构。
# 其与标准 GPT2 主要改变：
# 去除并停用了原始的绝对位置编码层 (wpe)。在 Attention 计算的 Q (Query) 和 K (Key) 的表征中，
# 直接注入了旋转位置嵌入(RoPE)，以引入相对位置信息。被认为比 relative embeddings 训练更快。
# ---------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0) # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0) # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1) # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1) # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GPTRotAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # 预先缓存一个够长的 position_ids 以避免每次 forward 都调用 torch.arange
        # 这样能略微提升一些前向传播效率
        self.register_buffer(
            "cached_position_ids", 
            torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0),
            persistent=False
        )

    def forward(self, hidden_states, past_key_values=None, attention_mask=None, position_ids=None, **kwargs):
        # 取代了标准架构 Q 和 K 的单纯点积：
        # 将输入拆分为 Q、K、V 后，将 RoPE (旋转编码的sin和cos计算分量) 应用于 Q 和 K。
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_kv).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if position_ids is None:
            # 直接从预先注册好的 buffer 里切片截取所需长度，省去动态 arange 带来的 host-to-device 调度开销
            seq_len = query_states.shape[2]
            position_ids = self.cached_position_ids[:, :seq_len]

        cos, sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached
        # Q 和 K 融合了旋转位置信息
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 随后的注意力矩阵计算等同于标准 dot product 注意力，但现在已具备了相对位置感知
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return self.c_proj(attn_output), attn_weights

# ---------------------------------------------------------
# GPT_rel 与 GPT_pos (基于 DeBERTa 框架的相对位置注意力)
#
# 对应论文中的 GPT_rel 与 GPT_pos 架构，这两者通过 mode='rel' / 'pos' 区分：
# 共通改变：去除了标准的绝对位置编码层。增加了一个可供训练的相对位置偏置 (Relative Position Bias)。
#
# GPT_rel (mode='rel'):
# - 标准注意力机制下附加该可供学习的“相对距离位置嵌入(|j - i|)”。这是在算好了 Q @ K^T 之后，加上偏置矩阵。
# 
# GPT_pos (mode='pos', Position-Only 注意力):
# - 它的 Attention 矩阵 **仅由位置决定**。
# - 直接省去 Q @ K^T 的点积计算（与 input 和 hidden states 无关），转而纯依靠相对位置的 Bias 进行 softmax 计算。
# - 是为了探索极限削弱版（无内容感知）的情况下单纯位置序列的推断能力。
# ---------------------------------------------------------
class GPT_Rel_Pos_Attention(GPT2Attention):
    def __init__(self, config, mode='rel', layer_idx=None):
        super().__init__(config, is_cross_attention=False, layer_idx=layer_idx)
        self.mode = mode # 'rel' or 'pos'
        self.relative_attention_bias = nn.Embedding(config.max_position_embeddings * 2, self.num_heads)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        seq_length = hidden_states.size(1)
        if self.mode == 'rel':
            # GPT_rel：保持特征交互能力，执行 Q 与 K 的点积
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            query_states = query_states.view(shape_kv).transpose(1, 2)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        else:
            # GPT_pos (纯位置): 去除了输入序列的Query和Key计算！
            # 注意力矩阵不再与隐藏状态发生交互内容。
            _, _, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*value_states.shape[:-1], -1, self.head_dim)
            value_states = value_states.view(shape_kv).transpose(1, 2)
            # 初始化为空的全零注意力矩阵 (因为没有 Q和K 来计算特征内容相似度)
            attn_weights = torch.zeros(hidden_states.size(0), self.num_heads, seq_length, seq_length, device=hidden_states.device)

        # 构建距离矩阵 (|j - i|) 映射为学习好的相对位置偏差(Bias)
        context_position = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)[:, None]
        memory_position = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)[None, :]
        relative_position = memory_position - context_position
        relative_position = relative_position + self.config.max_position_embeddings
        rel_pos_bias = self.relative_attention_bias(relative_position).permute(2, 0, 1).unsqueeze(0)
        
        # 将结构/相对位置信息加上注意力矩阵，供最后 softmax
        attn_weights = attn_weights + rel_pos_bias

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return self.c_proj(attn_output), attn_weights

# ---------------------------------------------------------
# GPT_uni (Uniform position-based attention)
# 对应论文中的 GPT_uni 架构。
#
# 最大改变和其代表的含义：
# 1. Attention 矩阵是被完全固定的（不可训练且与输入序列无关），只应用最简的位置注意力。
# 2. 这里的 h 头，不去做特征上的注意力投射 (无 Q 和 K 参数），只保留 Value 参数和输出投影。
# 3. 每 h 头对过去 (2^h - 1) 个 Tokens 执行一致均匀的权重平均 (Uniform Average)。
# ---------------------------------------------------------
class GPTUniAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.num_heads = config.num_attention_heads # e.g. 论文中为 8 头
        self.embed_dim = config.hidden_size # 论文中指出：较小的保持参数一致时是 840维，较大设定是 1024维
        self.head_dim = self.embed_dim // self.num_heads
        
        # 极简设计：不需要提取特征的自注意力权重，只执行Value线性投射与输出整合。
        self.c_fc = nn.Linear(self.embed_dim, self.embed_dim) 
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        bsz, seq_len, _ = hidden_states.shape
        value_states = self.c_fc(hidden_states)
        value_states = value_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 建立固定的 Uniform 注意力矩阵（恒定不可变）
        attn_weights = torch.zeros(1, self.num_heads, seq_len, seq_len, device=hidden_states.device)
        for h in range(self.num_heads):
            # 取以前的 2^h - 1 个Token窗口（依据论文的指数增长型上下文衰减）
            window_size = (2 ** (h + 1)) - 1
            for i in range(seq_len):
                start = max(0, i - window_size + 1)
                valid_len = i - start + 1
                # 执行均匀平均 (1.0 / window)
                attn_weights[0, h, i, start:i+1] = 1.0 / valid_len
                
        if attention_mask is not None:
            # Mask out padding if any
            attn_weights = attn_weights.masked_fill(attention_mask < -1e4, 0.0)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)

        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.c_proj(attn_output), attn_weights

# ---------------------------------------------------------
# Factory function for creating GPT blocks based on type
# ---------------------------------------------------------
def create_custom_gpt_block(config, layer_idx, gpt_type='standard'):
    block = GPT2Block(config, layer_idx=layer_idx)
    if gpt_type == 'rot':
        block.attn = GPTRotAttention(config, layer_idx=layer_idx)
    elif gpt_type == 'rel':
        block.attn = GPT_Rel_Pos_Attention(config, mode='rel', layer_idx=layer_idx)
    elif gpt_type == 'pos':
        block.attn = GPT_Rel_Pos_Attention(config, mode='pos', layer_idx=layer_idx)
    elif gpt_type == 'uni':
        block.attn = GPTUniAttention(config, layer_idx=layer_idx)
    return block

class DummyWPE(nn.Module):
    """一个不产生任何特征影响的伪装绝对位置编码层"""
    def forward(self, position_ids):
        # 返回形状为 scala 的 0.0，其可以被无损广播 (Broadcast) 并叠加到 inputs_embeds 上
        # 且相较于 Python 级别的函数截断会保持 Torch 静态计算图连贯且性能拉满。
        return torch.tensor(0.0, device=position_ids.device)

class CustomGPT2Model(GPT2Model):
    """
    通用 GPT 集成入口。
    gpt_type 参数选择包括: 
      - 'standard': 原始 GPT-2 / DeBERTa (带有绝对位置编码，或保持原样)
      - 'rot'     : GPT_rot (旋转位置编码)
      - 'rel'     : GPT_rel (基于内容的 DeBERTa 式相对位置偏置注意力)
      - 'pos'     : GPT_pos (仅包含位置偏置信息，无 Q-K 乘积的自注意力)
      - 'uni'     : GPT_uni (完全固定并取移动窗口指数平均的纯位置注意力)
      - 'rand'    : 如需实现 GPT_rand，调用 'rel'，并在训练过程中冻结模型使其不发生梯度下降，发挥随机构建 Kernel 特征的能力。
    """
    def __init__(self, config, gpt_type='standard'):
        super().__init__(config)
        self.gpt_type = gpt_type
        # 根据选定的 gpt_type 改写原始模型中的 Transformer Block 注意力替换实现
        self.h = nn.ModuleList([create_custom_gpt_block(config, i, gpt_type) for i in range(config.num_hidden_layers)])
        
        if gpt_type in ['rot', 'rel', 'pos', 'uni']:
            # 这些类型要么使用的是 RoPE，要么是通过 Relative Position Bias。
            # 直接替换为一个返回全零特征的静态 DummyWPE 并舍弃原本的 Embedding 层，大幅提高执行效率并清除计算图冗余
            self.wpe = DummyWPE()

from modeling_gpt2 import GPT2LMHeadModel

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    """
    用于带有语言模型（LM Head）任务预训练的模型封装。
    """
    def __init__(self, config, gpt_type='standard'):
        super().__init__(config)
        self.gpt_type = gpt_type
        # 抛弃预设的 GPT2Model 实例，将其替换为包含变体注意力的定制骨干网络
        self.transformer = CustomGPT2Model(config, gpt_type=gpt_type)
        # 二次执行 post_init 将其权重重新随机初始化，并确保 Embedding 与新实例化的 transformer 权重重新绑定同步
        self.post_init()
