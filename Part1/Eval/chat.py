import argparse
import sys

import torch


def extract_logits(outputs):
	return outputs.logits if hasattr(outputs, "logits") else outputs[0]


def main():
	parser = argparse.ArgumentParser(description="极简自回归生成（带 KV cache，逐步打印 token_id）")
	parser.add_argument("--model_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/GPT_2_Init/GPT_2_standard")
	parser.add_argument("--max_new_tokens", type=int, default=512)
	parser.add_argument("--temperature", type=float, default=0.0, help="0 表示 greedy")
	parser.add_argument("--bos_token", type=int, default=100)
	parser.add_argument("--eos_token", type=int, default=101)
	args = parser.parse_args()

	sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/model")
	from modeling_gpt2_variants import CustomGPT2LMHeadModel

	model = CustomGPT2LMHeadModel.from_pretrained(
		args.model_path,
		gpt_type='standard',
		device_map="auto",
	)
	model.eval()

	device = model.device

	input_ids = torch.tensor([[args.bos_token]], dtype=torch.long, device=device)
	generated = [args.bos_token]

	print(f"prompt(input_ids): {generated}")

	with torch.inference_mode():
		print("input_ids", input_ids)
		outputs = model(input_ids, use_cache=True)
		print("outputs", outputs)
		logits = extract_logits(outputs)
		print("logits", logits)
		print("logits shape", logits.shape)

		past_key_values = outputs.past_key_values

		for step in range(args.max_new_tokens):
			next_token_logits = logits[0, -1, :]

			if args.temperature > 0:
				probs = torch.softmax(next_token_logits / args.temperature, dim=-1)
				print("probs", probs)
				next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
			else:
				next_token = torch.argmax(next_token_logits).view(1, 1)
				print("next_token", next_token)

			token_id = int(next_token.item())
			generated.append(token_id)
			print(f"step={step + 1:03d} token_id={token_id}")

			if token_id == args.eos_token:
				print("hit EOS, stop")
				break

			outputs = model(next_token, use_cache=True, past_key_values=past_key_values)
			logits = extract_logits(outputs)
			past_key_values = outputs.past_key_values

	print(f"final_ids: {generated}")


if __name__ == "__main__":
    main()
