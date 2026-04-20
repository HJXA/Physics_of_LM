from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/ruilab/jxhe/CoE_Monitor/checkpoints/llama-2-7b")

def init_and_save(name, config, save_path):
    model = LlamaForCausalLM(config)
    total_params = model.num_parameters()
    if total_params >= 1e9:
        print(f"{name}参数量: {total_params / 1e9:.2f} B")
    else:
        print(f"{name}参数量: {total_params / 1e6:.2f} M")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"{name}已保存到: {save_path}")


# ========== 162M 参数模型 ==========
init_and_save("162M", LlamaConfig(
    hidden_size=768,
    intermediate_size=768*4,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
), "./Part3/checkpoints/llama2_162M")

# ========== 83M 参数模型 ==========
init_and_save("83M", LlamaConfig(
    hidden_size=512,
    intermediate_size=512*4,
    num_hidden_layers=12,
    num_attention_heads=8,
    max_position_embeddings=512,
), "./Part3/checkpoints/llama2_83m")

# ========== 40M 参数模型 ==========
init_and_save("40M", LlamaConfig(
    hidden_size=384,
    intermediate_size=384*4,
    num_hidden_layers=6,
    num_attention_heads=6,
    max_position_embeddings=512,
), "./Part3/checkpoints/llama2_40m")

# ========== 20M 参数模型 ==========
init_and_save("20M", LlamaConfig(
    hidden_size=256,
    intermediate_size=256*4,
    num_hidden_layers=6,
    num_attention_heads=4,
    max_position_embeddings=512,
), "./Part3/checkpoints/llama2_20m")

# ========== 10M 参数模型 ==========
init_and_save("10M", LlamaConfig(
    hidden_size=128,
    intermediate_size=128*4,
    num_hidden_layers=6,
    num_attention_heads=4,
    max_position_embeddings=512,
), "./Part3/checkpoints/llama2_10m")
