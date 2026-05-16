import argparse

import torch
from transformers import AutoTokenizer

from model import ModelConfig, Transformer


def load_model(checkpoint_path: str, tokenizer, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_dict = checkpoint.get("config", {})

    config = ModelConfig(
        vocab_size=len(tokenizer),
        dim=config_dict.get("dim", 512),
        n_layers=config_dict.get("n_layers", 8),
        n_heads=config_dict.get("n_heads", 8),
        n_kv_heads=config_dict.get("n_kv_heads", None),
        max_seq_len=config_dict.get("max_seq_len", 512),
        dropout=0.0,
    )

    model = Transformer(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model

def generate_pretrain(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
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

    generated_ids = output_ids[0].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    return text

def generate_sft(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": prompt},
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_ids = output_ids[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return text.strip()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")

    parser.add_argument("--mode", type=str, choices=["pretrain", "sft"], default="sft")
    parser.add_argument("--prompt", type=str, default="你好")

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        device=device,
    )

    if args.mode == "pretrain":
        text = generate_pretrain(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    else:
        text = generate_sft(
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
