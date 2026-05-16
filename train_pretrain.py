import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from dataset import PretrainDataset
from model import ModelConfig, Transformer


def setup_distributed():
    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if not is_ddp:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "is_ddp": False,
            "device": device,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main": True,
        }

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = f"cuda:{local_rank}"
    else:
        backend = "gloo"
        device = "cpu"

    dist.init_process_group(backend=backend)

    return {
        "is_ddp": True,
        "device": device,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def cleanup_distributed(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


def print_main(is_main: bool, *args, **kwargs):
    if is_main:
        print(*args, **kwargs)


def get_lr(step: int, total_steps: int, learning_rate: float, warmup_steps: int):
    if step < warmup_steps:
        return learning_rate * step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    return 0.5 * learning_rate * (1.0 + math.cos(math.pi * progress))


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(model, optimizer, config, args, step: int, out_dir: Path, is_main: bool):
    if not is_main:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.__dict__,
        "args": vars(args),
        "step": step,
    }

    path = out_dir / f"pretrain_step_{step}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, resume_path: str, device: str):
    if not resume_path:
        return 0

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return int(checkpoint.get("step", 0))


def reduce_mean(tensor: torch.Tensor, is_ddp: bool):
    if not is_ddp:
        return tensor

    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def train(args):
    ddp = setup_distributed()
    device = ddp["device"]
    is_ddp = ddp["is_ddp"]
    is_main = ddp["is_main"]
    world_size = ddp["world_size"]

    torch.manual_seed(args.seed + ddp["rank"])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    config = ModelConfig(
        vocab_size=len(tokenizer),
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    dataset = PretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=ddp["rank"],
        shuffle=True,
    ) if is_ddp else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = Transformer(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    global_step = load_checkpoint(model, optimizer, args.resume, device)

    if is_ddp:
        model = DDP(
            model,
            device_ids=[ddp["local_rank"]] if device.startswith("cuda") else None,
        )

    total_steps = args.epochs * len(dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    print_main(is_main, f"Device: {device}")
    print_main(is_main, f"DDP: {is_ddp}, world size: {world_size}")
    print_main(is_main, f"Dataset size: {len(dataset)}")
    print_main(is_main, f"Steps per epoch: {len(dataloader)}")
    print_main(is_main, f"Total steps: {total_steps}")
    print_main(is_main, f"Warmup steps: {warmup_steps}")
    print_main(is_main, f"Resume step: {global_step}")
    print_main(
        is_main,
        f"Model parameters: {sum(p.numel() for p in unwrap_model(model).parameters()) / 1e6:.2f}M",
    )

    start_step = global_step
    start_epoch = start_step // max(1, len(dataloader))
    step_in_start_epoch = start_step % max(1, len(dataloader))

    model.train()
    start_time = time.time()
    last_log_time = start_time

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            if epoch == start_epoch and batch_idx < step_in_start_epoch:
                continue

            global_step += 1

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            lr = get_lr(
                step=global_step,
                total_steps=total_steps,
                learning_rate=args.learning_rate,
                warmup_steps=warmup_steps,
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            logits, loss = model(input_ids, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            if global_step % args.log_interval == 0:
                now = time.time()
                elapsed = time.time() - start_time
                interval = max(1e-6, now - last_log_time)
                tokens_per_step = input_ids.numel() * world_size
                tokens_per_sec = tokens_per_step * args.log_interval / interval
                last_log_time = now
                mean_loss = reduce_mean(loss, is_ddp)

                print_main(
                    is_main,
                    f"epoch {epoch + 1} "
                    f"step {global_step}/{total_steps} "
                    f"loss {mean_loss.item():.4f} "
                    f"lr {lr:.6e} "
                    f"tokens/s {tokens_per_sec:.0f} "
                    f"time {elapsed:.1f}s"
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(model, optimizer, config, args, global_step, Path(args.out_dir), is_main)

    save_checkpoint(model, optimizer, config, args, global_step, Path(args.out_dir), is_main)
    cleanup_distributed(is_ddp)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/raw/sample.txt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--out_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=128)

    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
