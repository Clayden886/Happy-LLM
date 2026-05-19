import argparse
import math
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from dataset import PretrainDataset
from model import DecoderLayer, ModelConfig, Transformer


def setup_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        raise RuntimeError("Launch FSDP training with torchrun and at least 2 processes.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("This FSDP training script expects CUDA GPUs.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return {
        "device": torch.device("cuda", local_rank),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def cleanup_distributed():
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


def reduce_mean(tensor: torch.Tensor):
    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def apply_checkpointing(model: torch.nn.Module):
    wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, DecoderLayer),
    )


def load_model_weights(model, checkpoint_path: str, device: torch.device, strict: bool = True):
    if not checkpoint_path:
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0


def save_checkpoint(model, optimizer, config, args, step: int, out_dir: Path, is_main: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    state_policy = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    optim_policy = FullOptimStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_policy,
        optim_policy,
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer) if args.save_optimizer else None

    if not is_main:
        return

    checkpoint = {
        "model": model_state,
        "config": config.__dict__,
        "args": vars(args),
        "step": step,
    }
    if optim_state is not None:
        checkpoint["optimizer"] = optim_state

    path = out_dir / f"pretrain_step_{step}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def train(args):
    ddp = setup_distributed()
    device = ddp["device"]
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
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Transformer(config)
    total_params = count_parameters(model)
    loaded_step = load_model_weights(model, args.init_checkpoint, device)

    if args.activation_checkpointing:
        apply_checkpointing(model)

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    global_step = loaded_step

    print_main(is_main, f"Device: {device}")
    print_main(is_main, f"FSDP world size: {world_size}")
    print_main(is_main, f"Dataset size: {len(dataset)}")
    print_main(is_main, f"Micro steps per epoch: {len(dataloader)}")
    print_main(is_main, f"Optimizer steps per epoch: {steps_per_epoch}")
    print_main(is_main, f"Total optimizer steps: {total_steps}")
    print_main(is_main, f"Warmup steps: {warmup_steps}")
    print_main(is_main, f"Init checkpoint step: {loaded_step}")
    print_main(is_main, f"Gradient accumulation steps: {args.grad_accum_steps}")
    print_main(is_main, f"Activation checkpointing: {args.activation_checkpointing}")
    print_main(is_main, f"Save optimizer state: {args.save_optimizer}")
    print_main(is_main, f"Model parameters: {total_params / 1e6:.2f}M")

    model.train()
    start_time = time.time()
    last_log_time = start_time
    running_loss = torch.tensor(0.0, device=device)
    accum_count = 0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            is_accum_boundary = (accum_count + 1) % args.grad_accum_steps == 0
            is_last_micro_step = batch_idx + 1 == len(dataloader)
            should_step = is_accum_boundary or is_last_micro_step

            use_no_sync = args.no_sync_grad_accum and not should_step
            sync_context = model.no_sync() if use_no_sync else nullcontext()
            with sync_context:
                _, loss = model(input_ids, labels)
                (loss / args.grad_accum_steps).backward()

            running_loss += loss.detach()
            accum_count += 1

            if not should_step:
                continue

            global_step += 1
            lr = get_lr(global_step, total_steps, args.learning_rate, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if args.grad_clip > 0:
                model.clip_grad_norm_(args.grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_interval == 0:
                now = time.time()
                elapsed = now - start_time
                interval = max(1e-6, now - last_log_time)
                tokens_per_step = input_ids.numel() * world_size * args.grad_accum_steps
                tokens_per_sec = tokens_per_step * args.log_interval / interval
                last_log_time = now

                mean_loss = reduce_mean(running_loss / max(1, accum_count))
                running_loss.zero_()
                accum_count = 0

                print_main(
                    is_main,
                    f"epoch {epoch + 1} "
                    f"step {global_step}/{total_steps} "
                    f"loss {mean_loss.item():.4f} "
                    f"lr {lr:.6e} "
                    f"tokens/s {tokens_per_sec:.0f} "
                    f"time {elapsed:.1f}s",
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(model, optimizer, config, args, global_step, Path(args.out_dir), is_main)
                dist.barrier()

    save_checkpoint(model, optimizer, config, args, global_step, Path(args.out_dir), is_main)
    dist.barrier()
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/pretrain_zhwiki_simplified.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--out_dir", type=str, default="checkpoints/pretrain_500m")
    parser.add_argument("--init_checkpoint", type=str, default="")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--n_layers", type=int, default=18)
    parser.add_argument("--n_heads", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--no_sync_grad_accum", action="store_true")
    parser.add_argument("--save_optimizer", action="store_true")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
