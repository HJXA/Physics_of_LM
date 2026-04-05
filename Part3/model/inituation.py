from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

# 1️⃣ 构建 config
config = LlamaConfig(
    hidden_size=768,
    intermediate_size=768*4,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)

tokenizer = AutoTokenizer.from_pretrained("/ruilab/jxhe/CoE_Monitor/checkpoints/llama-2-7b")

# 2️⃣ 初始化模型（随机权重）
model = LlamaForCausalLM(config)

total_params = model.num_parameters()

if total_params >= 1e9:
    print(f"参数量: {total_params / 1e9:.2f} B")
else:
    print(f"参数量: {total_params / 1e6:.2f} M")

# 3️⃣ 保存路径
save_path = "./Part3/checkpoints/llama2_init"

# 4️⃣ 保存模型 + config
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"模型已保存到: {save_path}")