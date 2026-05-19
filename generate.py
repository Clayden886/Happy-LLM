import argparse

import torch
from transformers import AutoTokenizer

from model import ModelConfig, Transformer


def load_model(checkpoint_path: str, tokenizer, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get("config", {})

    config = ModelConfig(
        vocab_size=len(tokenizer),
        dim=config_dict.get("dim", 1536),
        n_layers=config_dict.get("n_layers", 16),
        n_heads=config_dict.get("n_heads", 24),
        n_kv_heads=config_dict.get("n_kv_heads", None),
        max_seq_len=config_dict.get("max_seq_len", 512),
        dropout=0.0,
    )

    model = Transformer(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--prompt", type=str, default="人工智能")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.checkpoint, tokenizer, device)

    input_ids = tokenizer.encode(
        args.prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
