import argparse
import json
import os
from pathlib import Path


SYSTEM_PROMPT = "你是一个AI助手。"


def get_load_dataset():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install project dependencies with "
            "`pip install -r requirements.txt` before preparing datasets."
        ) from exc

    return load_dataset


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def iter_pretrain_text(
    dataset_name: str,
    split: str,
    text_field: str,
    min_chars: int,
    max_docs: int,
    max_bytes: int,
    token: str | None,
):
    load_dataset = get_load_dataset()
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        token=token,
    )

    seen_docs = 0
    seen_bytes = 0

    for item in dataset:
        text = clean_text(str(item.get(text_field, "")))
        if len(text) < min_chars:
            continue

        row = {"text": text}
        row_bytes = len(json.dumps(row, ensure_ascii=False).encode("utf-8")) + 1

        if max_docs > 0 and seen_docs >= max_docs:
            break
        if max_bytes > 0 and seen_bytes + row_bytes > max_bytes:
            break

        seen_docs += 1
        seen_bytes += row_bytes
        yield row


def prepare_pretrain(args):
    token = args.hf_token or os.environ.get("HF_TOKEN")
    rows = iter_pretrain_text(
        dataset_name=args.dataset_name,
        split=args.split,
        text_field=args.text_field,
        min_chars=args.min_chars,
        max_docs=args.max_docs,
        max_bytes=args.max_bytes,
        token=token,
    )

    count = write_jsonl(Path(args.out_path), rows)
    print(f"Saved {count} pretrain samples to {args.out_path}")


def build_user_content(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()

    if input_text:
        return f"{instruction}\n\n{input_text}"

    return instruction


def iter_alpaca_zh(split: str, max_samples: int):
    load_dataset = get_load_dataset()
    dataset = load_dataset("hfl/alpaca_zh_51k", split=split)

    count = 0
    for item in dataset:
        instruction = str(item.get("instruction", "")).strip()
        input_text = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()

        if not instruction or not output:
            continue

        if max_samples > 0 and count >= max_samples:
            break

        count += 1
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(instruction, input_text)},
                {"role": "assistant", "content": output},
            ]
        }


def prepare_sft(args):
    rows = iter_alpaca_zh(
        split=args.split,
        max_samples=args.max_samples,
    )

    count = write_jsonl(Path(args.out_path), rows)
    print(f"Saved {count} SFT samples to {args.out_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain-cci3-hq")
    pretrain.add_argument("--dataset_name", type=str, default="BAAI/CCI3-HQ")
    pretrain.add_argument("--split", type=str, default="train")
    pretrain.add_argument("--text_field", type=str, default="text")
    pretrain.add_argument("--out_path", type=str, default="data/processed/pretrain_cci3_hq.jsonl")
    pretrain.add_argument("--min_chars", type=int, default=80)
    pretrain.add_argument("--max_docs", type=int, default=10000)
    pretrain.add_argument("--max_bytes", type=int, default=0)
    pretrain.add_argument("--hf_token", type=str, default="")
    pretrain.set_defaults(func=prepare_pretrain)

    sft = subparsers.add_parser("sft-alpaca-zh")
    sft.add_argument("--split", type=str, default="train")
    sft.add_argument("--out_path", type=str, default="data/processed/sft_alpaca_zh_51k.jsonl")
    sft.add_argument("--max_samples", type=int, default=0)
    sft.set_defaults(func=prepare_sft)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
