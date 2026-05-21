# Happy LLM

一个面向 Tesla M40 24GB 服务器的中文 Decoder-only Transformer 项目。当前主线回到更稳定的 DDP：支持中文预训练、SFT 微调和续写/问答测试。

## Structure

```text
.
├── model.py                    # LLaMA-like Transformer: RMSNorm / RoPE / SwiGLU
├── dataset.py                  # PretrainDataset / SFTDataset
├── train_pretrain.py           # DDP 预训练入口
├── train_sft.py                # DDP SFT 微调入口
├── generate.py                 # 预训练续写 / SFT 问答测试
├── requirements.txt            # 非 PyTorch 依赖
├── scripts/
│   ├── prepare_zhwiki.py       # WikiExtractor 输出转预训练 JSONL
│   ├── build_pretrain_corpus.py # 合并多个中文语料为统一 JSONL
│   └── train_tokenizer.py      # BPE tokenizer 训练
├── data/
│   ├── raw/                    # 原始数据，不提交
│   └── processed/              # 处理后数据，不提交
├── tokenizer/                  # tokenizer 产物，不提交
└── checkpoints/                # checkpoint，不提交
```

## Environment

推荐 Python 3.10。Tesla M40 建议使用 CUDA 11.8 对应的 PyTorch。

```bash
conda create -p /data2/tangyb/conda_envs/happy-llm python=3.10 -y
conda activate /data2/tangyb/conda_envs/happy-llm

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

项目默认使用 FP32，不依赖 FlashAttention、xFormers 或 bitsandbytes。

## Data

预训练数据格式：

```jsonl
{"text": "这里是一段中文文本。"}
```

SFT 数据格式：

```jsonl
{"messages": [{"role": "system", "content": "你是一个AI助手。"}, {"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你？"}]}
```

准备中文维基：

```bash
python -m wikiextractor.WikiExtractor \
  data/raw/zhwiki/zhwiki-latest-pages-articles.xml.bz2 \
  -o data/raw/zhwiki/extracted \
  --json

python scripts/prepare_zhwiki.py \
  --input_dir data/raw/zhwiki/extracted \
  --out_path data/processed/pretrain_zhwiki_simplified.jsonl \
  --min_chars 80 \
  --t2s
```

合并多个预训练语料：

```bash
python scripts/build_pretrain_corpus.py \
  --inputs data/raw/seq_monkey data/processed/pretrain_zhwiki_simplified.jsonl \
  --out_path data/processed/pretrain_all.jsonl \
  --min_chars 80 \
  --t2s
```

## Tokenizer

建议用 2GB-5GB 代表性语料训练 tokenizer，不必吃完整 33GB 预训练数据。

```bash
python scripts/build_pretrain_corpus.py \
  --inputs data/processed/pretrain_all.jsonl \
  --out_path data/tokenizer_train/pretrain_tokenizer_5gb.jsonl \
  --min_chars 80 \
  --max_bytes 5GB

rm -rf tokenizer

python scripts/train_tokenizer.py \
  --data_dir data/tokenizer_train \
  --out_dir tokenizer \
  --vocab_size 12000 \
  --model_max_length 512
```

## Pretrain With DDP

M40 双卡更适合 DDP 训练 90M-200M 级模型。推荐先跑 160M：

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=4,5 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=2 train_pretrain.py \
  --data_path data/processed/pretrain_all.jsonl \
  --tokenizer_path tokenizer \
  --out_dir checkpoints/pretrain_160m \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --max_seq_len 512 \
  --dim 1024 \
  --n_layers 12 \
  --n_heads 16 \
  --learning_rate 3e-4 \
  --log_interval 20 \
  --save_interval 1000 \
  2>&1 | tee logs/pretrain_160m.log
```

如果稳定且显存足够，可以尝试 210M：

```bash
--dim 1024 --n_layers 16 --n_heads 16
```

## SFT

SFT 需要先有预训练 checkpoint。假设预训练结果为：

```text
checkpoints/pretrain_160m/pretrain_step_XXXXX.pt
```

运行 SFT：

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=4,5 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=2 train_sft.py \
  --data_path data/processed/sft_alpaca_chinese_20k.jsonl \
  --tokenizer_path tokenizer \
  --out_dir checkpoints/sft_160m \
  --pretrained_checkpoint checkpoints/pretrain_160m/pretrain_step_XXXXX.pt \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --max_seq_len 512 \
  --dim 1024 \
  --n_layers 12 \
  --n_heads 16 \
  --learning_rate 2e-5 \
  --log_interval 20 \
  --save_interval 1000 \
  2>&1 | tee logs/sft_160m.log
```

## Generate

预训练模型续写：

```bash
CUDA_VISIBLE_DEVICES=4 python generate.py \
  --checkpoint checkpoints/pretrain_160m/pretrain_step_XXXXX.pt \
  --tokenizer_path tokenizer \
  --mode pretrain \
  --prompt "人工智能是" \
  --max_new_tokens 120 \
  --temperature 0.8 \
  --top_k 50
```

SFT 模型问答：

```bash
CUDA_VISIBLE_DEVICES=4 python generate.py \
  --checkpoint checkpoints/sft_160m/sft_step_XXXXX.pt \
  --tokenizer_path tokenizer \
  --mode sft \
  --prompt "请用三句话解释什么是大语言模型。" \
  --max_new_tokens 160 \
  --temperature 0.3 \
  --top_k 10
```

## Notes For M40

- 优先使用已验证稳定的 GPU 组合，例如 `CUDA_VISIBLE_DEVICES=4,5`。
- 避开有 ECC double-bit 记录或残留异常进程的 GPU。
- 长任务放在 `tmux` 中运行，并用 `tee` 保存日志。
- 这台机器上 FSDP 容易触发 timeout，当前推荐 DDP 主线。
