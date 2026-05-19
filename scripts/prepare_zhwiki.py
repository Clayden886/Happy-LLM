import argparse
import json
from pathlib import Path


def get_converter(enabled: bool):
    if not enabled:
        return lambda text: text

    try:
        from opencc import OpenCC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: opencc-python-reimplemented. "
            "Install with `pip install -r requirements.txt`."
        ) from exc

    converter = OpenCC("t2s")
    return converter.convert


def clean_text(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def iter_wikiextractor_json(input_dir: Path):
    for path in sorted(input_dir.rglob("wiki_*")):
        if not path.is_file():
            continue

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw/zhwiki/extracted")
    parser.add_argument("--out_path", type=str, default="data/processed/pretrain_zhwiki_simplified.jsonl")
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--t2s", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    convert = get_converter(args.t2s)
    count = 0

    with out_path.open("w", encoding="utf-8") as w:
        for item in iter_wikiextractor_json(input_dir):
            text = clean_text(str(item.get("text", "")))
            text = convert(text)

            if len(text) < args.min_chars:
                continue

            w.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1

            if args.max_docs > 0 and count >= args.max_docs:
                break

    print(f"Saved {count} documents to {out_path}")


if __name__ == "__main__":
    main()
