import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "<unk>",
    "<s>",
    "</s>",
    "<|im_start|>",
    "<|im_end|>",
]


def iter_text_files(data_dir: Path):
    for path in sorted(data_dir.glob("*")):
        if path.is_file() and path.suffix in {".txt", ".jsonl"}:
            yield path


def iter_texts(data_dir: Path):
    for path in iter_text_files(data_dir):
        if path.suffix == ".txt":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        yield text

        elif path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)
                    text = str(item.get("text", "")).strip()
                    if text:
                        yield text


def train_tokenizer(data_dir: Path, out_dir: Path, vocab_size: int, model_max_length: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(iter_texts(data_dir), trainer=trainer)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    fast_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    )
    fast_tokenizer.model_max_length = model_max_length
    fast_tokenizer.save_pretrained(str(out_dir))

    test_text = "你好，我正在从零开始训练一个五亿参数中文语言模型。"
    ids = fast_tokenizer.encode(test_text)

    print(f"Tokenizer saved to: {out_dir}")
    print(f"Vocab size: {fast_tokenizer.vocab_size}")
    print("Test text:", test_text)
    print("Token ids:", ids)
    print("Decoded:", fast_tokenizer.decode(ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="tokenizer")
    parser.add_argument("--vocab_size", type=int, default=12000)
    parser.add_argument("--model_max_length", type=int, default=512)
    args = parser.parse_args()

    train_tokenizer(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        vocab_size=args.vocab_size,
        model_max_length=args.model_max_length,
    )


if __name__ == "__main__":
    main()
