import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()

    def _load_samples(self) -> List[str]:
        samples = []

        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if self.data_path.suffix == ".jsonl":
                    item = json.loads(line)
                    text = str(item.get("text", "")).strip()
                else:
                    text = line

                if text:
                    samples.append(text)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        input_ids = self.tokenizer.encode(
            self.samples[idx],
            add_special_tokens=False,
        )

        input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        input_ids = input_ids[: self.max_length + 1]

        padding_length = self.max_length + 1 - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        x = input_ids[:-1]
        y = input_ids[1:]
        y = y.masked_fill(y == self.tokenizer.pad_token_id, -100)

        return {
            "input_ids": x,
            "labels": y,
        }
