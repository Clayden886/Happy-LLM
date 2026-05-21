import argparse
import hashlib
import json
from pathlib import Path


TEXT_FIELDS = ("text", "content", "data", "正文")
SUPPORTED_SUFFIXES = (".jsonl", ".json", ".txt", ".parquet")


def get_converter(enabled: bool):
    if not enabled:
        return lambda text: text

    try:
        from opencc import OpenCC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: opencc-python-reimplemented. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc

    return OpenCC("t2s").convert


def clean_text(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def get_text(item):
    if isinstance(item, str):
        return item

    if not isinstance(item, dict):
        return ""

    for field in TEXT_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def iter_input_files(path: Path):
    if path.is_file():
        yield path
        return

    for suffix in SUPPORTED_SUFFIXES:
        yield from sorted(path.rglob(f"*{suffix}"))


def iter_records(path: Path):
    if path.suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                yield line
        return

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            for key in ("data", "train", "rows", "documents"):
                value = data.get(key)
                if isinstance(value, list):
                    yield from value
                    return
            yield data

        return

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: pyarrow. Install it with "
                "`pip install -r requirements.txt`."
            ) from exc

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=2048):
            columns = batch.to_pydict()
            row_count = len(next(iter(columns.values()), []))
            for index in range(row_count):
                yield {key: value[index] for key, value in columns.items()}


def parse_max_bytes(value: str):
    if not value:
        return 0

    value = value.strip().upper()
    units = {
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
    }

    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            number = float(value[: -len(suffix)])
            return int(number * multiplier)

    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files or directories containing .jsonl/.json/.txt/.parquet files.",
    )
    parser.add_argument("--out_path", type=str, default="data/processed/pretrain_all.jsonl")
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_bytes", type=str, default="")
    parser.add_argument("--t2s", action="store_true")
    parser.add_argument("--dedupe", action="store_true", help="Remove exact duplicate texts.")
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = parse_max_bytes(args.max_bytes)
    convert = get_converter(args.t2s)

    written = 0
    saved = 0
    skipped = 0
    duplicated = 0
    seen_hashes = set()

    with out_path.open("w", encoding="utf-8") as w:
        for input_arg in args.inputs:
            input_path = Path(input_arg)
            if not input_path.exists():
                print(f"skip missing: {input_path}")
                continue

            for path in iter_input_files(input_path):
                print(f"reading {path}")

                for item in iter_records(path):
                    text = convert(clean_text(get_text(item)))
                    if len(text) < args.min_chars:
                        skipped += 1
                        continue

                    if args.dedupe:
                        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
                        if digest in seen_hashes:
                            duplicated += 1
                            continue
                        seen_hashes.add(digest)

                    row = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    row_bytes = len(row.encode("utf-8"))

                    if max_bytes > 0 and written + row_bytes > max_bytes:
                        print("reached max_bytes")
                        print(f"saved docs: {saved}")
                        print(f"skipped: {skipped}")
                        print(f"duplicated: {duplicated}")
                        print(f"size GB: {written / 1024**3:.2f}")
                        print(f"output: {out_path}")
                        return

                    w.write(row)
                    written += row_bytes
                    saved += 1

                    if saved % 100000 == 0:
                        print(
                            f"saved {saved} docs, "
                            f"{written / 1024**3:.2f} GB, "
                            f"skipped {skipped}, "
                            f"duplicated {duplicated}"
                        )

    print("done")
    print(f"saved docs: {saved}")
    print(f"skipped: {skipped}")
    print(f"duplicated: {duplicated}")
    print(f"size GB: {written / 1024**3:.2f}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
