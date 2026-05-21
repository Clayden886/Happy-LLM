import argparse

import torch
from transformers import AutoTokenizer

from model import ModelConfig, Transformer


def load_model(checkpoint_path: str, tokenizer, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get("config", {})

    config = ModelConfig(
        vocab_size=len(tokenizer),
        dim=config_dict.get("dim", 1024),
        n_layers=config_dict.get("n_layers", 12),
        n_heads=config_dict.get("n_heads", 16),
        n_kv_heads=config_dict.get("n_kv_heads", None),
        max_seq_len=config_dict.get("max_seq_len", 512),
        dropout=0.0,
    )

    model = Transformer(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def generate_pretrain(model, tokenizer, prompt, device, max_new_tokens, temperature, top_k):
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)


def generate_sft(model, tokenizer, prompt, device, max_new_tokens, temperature, top_k):
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": prompt},
    ]

    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded.to(device) if isinstance(encoded, torch.Tensor) else encoded["input_ids"].to(device)
    else:
        prompt_text = (
            "<|im_start|>system\n你是一个AI助手。<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        input_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_ids = output_ids[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument("--prompt", type=str, default="人工智能")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.checkpoint, tokenizer, device)

    if args.mode == "sft":
        text = generate_sft(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    else:
        text = generate_pretrain(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(text)


if __name__ == "__main__":
    main()
