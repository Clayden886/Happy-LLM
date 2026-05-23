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
            truncation=True,
            max_length=max(1, self.max_length - 2),
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


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                if "messages" in item:
                    messages = item["messages"]
                else:
                    instruction = str(item.get("instruction", item.get("human", ""))).strip()
                    input_text = str(item.get("input", "")).strip()
                    output = str(item.get("output", item.get("assistant", ""))).strip()

                    if input_text:
                        instruction = f"{instruction}\n\n输入：\n{input_text}"

                    messages = [
                        {"role": "system", "content": "你是一个AI助手。"},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": output},
                    ]

                if messages:
                    samples.append(messages)

        return samples

    def __len__(self):
        return len(self.samples)

    def _build_prompt_and_labels(self, messages):
        input_ids = []
        labels = []

        for message in messages:
            role = message["role"]
            content = message["content"]
            message_text = f"<|im_start|>{role}\n{content}<|im_end|>\n"

            message_ids = self.tokenizer.encode(
                message_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length + 1,
            )

            input_ids.extend(message_ids)
            if role == "assistant":
                labels.extend(message_ids)
            else:
                labels.extend([-100] * len(message_ids))

        input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        labels = [-100] + labels + [self.tokenizer.eos_token_id]

        input_ids = input_ids[: self.max_length + 1]
        labels = labels[: self.max_length + 1]

        padding_length = self.max_length + 1 - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            labels += [-100] * padding_length

        return input_ids, labels

    def __getitem__(self, idx: int):
        input_ids, labels = self._build_prompt_and_labels(self.samples[idx])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids[:-1],
            "labels": labels[1:],
        }
