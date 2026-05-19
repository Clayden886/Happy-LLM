# Happy LLM 500M

面向 Tesla M40 24GB 双卡服务器的中文 Decoder-only Transformer 预训练项目。当前版本只保留 500M 级 FSDP 预训练所需的最小代码路径：中文维基数据处理、BPE tokenizer 训练、FSDP 预训练和续写测试。

## Structure

```text
.
├── model.py                 # LLaMA-like Transformer: RMSNorm / RoPE / SwiGLU
├── dataset.py               # 预训练 JSONL/TXT 数据集
├── train_pretrain.py        # FSDP 预训练入口
├── generate.py              # 预训练模型续写测试
├── requirements.txt         # 非 PyTorch 依赖
├── scripts/
│   ├── prepare_zhwiki.py    # WikiExtractor 输出转预训练 JSONL
│   ├── build_pretrain_corpus.py # 合并多个中文语料为统一 JSONL
│   └── train_tokenizer.py   # BPE tokenizer 训练
├── data/
│   ├── raw/                 # 原始数据，不提交
│   └── processed/           # 处理后数据，不提交
├── tokenizer/               # tokenizer 产物，不提交
└── checkpoints/             # checkpoint，不提交
```

## Environment

推荐 Python 3.10。Tesla M40 建议使用 CUDA 11.8 对应的 PyTorch，避免过新的 CUDA wheel 带来兼容风险。

```bash
conda create -p /data2/tangyb/conda_envs/happy-llm python=3.10 -y
conda activate /data2/tangyb/conda_envs/happy-llm

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

项目默认使用 FP32，不依赖 FlashAttention、xFormers、bitsandbytes 或 Tensor Core。

## Data

训练数据格式：

```jsonl
{"text": "这里是一段中文文本。"}
```

推荐使用中文维基百科 dump。先用 WikiExtractor 解包：

```bash
python -m wikiextractor.WikiExtractor \
  data/raw/zhwiki/zhwiki-latest-pages-articles.xml.bz2 \
  -o data/raw/zhwiki/extracted \
  --json
```

再转换成项目训练格式，并将繁体转简体：

```bash
python scripts/prepare_zhwiki.py \
  --input_dir data/raw/zhwiki/extracted \
  --out_path data/processed/pretrain_zhwiki_simplified.jsonl \
  --min_chars 80 \
  --t2s
```

检查数据：

```bash
wc -l data/processed/pretrain_zhwiki_simplified.jsonl
du -sh data/processed/pretrain_zhwiki_simplified.jsonl
head -n 1 data/processed/pretrain_zhwiki_simplified.jsonl
```

如果还下载了其他中文 JSONL/TXT 语料，可以合并为一个训练文件。脚本会自动读取 `.jsonl` 和 `.txt`，从常见字段 `text/content/data/正文` 中取正文，并统一输出为 `{"text": "..."}`：

```bash
python scripts/build_pretrain_corpus.py \
  --inputs \
    data/raw/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
    data/processed/pretrain_zhwiki_simplified.jsonl \
  --out_path data/processed/pretrain_all.jsonl \
  --min_chars 80 \
  --t2s
```

如果想临时限制数据大小，例如只构建 50GB：

```bash
python scripts/build_pretrain_corpus.py \
  --inputs data/raw/seq_monkey data/processed/pretrain_zhwiki_simplified.jsonl \
  --out_path data/processed/pretrain_50gb.jsonl \
  --max_bytes 50GB \
  --t2s
```

## Tokenizer

500M 级模型默认使用 12000 词表，比早期 6144 词表更适合中文维基语料。

```bash
rm -rf tokenizer

python scripts/train_tokenizer.py \
  --data_dir data/processed \
  --out_dir tokenizer \
  --vocab_size 12000 \
  --model_max_length 512
```

训练完成后确认没有大量 `<unk>`：

```bash
python - << 'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tokenizer")
text = "你好，我正在训练一个五亿参数中文语言模型。"
ids = tok.encode(text)
print("vocab:", len(tok))
print(ids)
print(tok.decode(ids))
PY
```

## Train 500M

默认配置：

```text
dim = 1536
n_layers = 18
n_heads = 24
max_seq_len = 512
vocab_size ≈ 12000
parameters ≈ 528M
```

建议先使用已经验证较稳定的 GPU4、GPU5。M40 没有 NVLink，FSDP 通信会比较慢，所以脚本默认保显存优先：梯度累积时每个 micro step 都同步并切分梯度。确认显存充足后，可以额外加 `--no_sync_grad_accum` 减少通信。

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=4,5 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=2 train_pretrain.py \
  --data_path data/processed/pretrain_all.jsonl \
  --tokenizer_path tokenizer \
  --out_dir checkpoints/pretrain_500m \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --max_seq_len 512 \
  --dim 1536 \
  --n_layers 18 \
  --n_heads 24 \
  --activation_checkpointing \
  --learning_rate 1e-4 \
  --log_interval 10 \
  --save_interval 500 \
  2>&1 | tee logs/pretrain_500m.log
```

如果 528M 显存或通信压力太大，先降到 462M：

```bash
--n_layers 16
```

Checkpoint 默认只保存模型权重、配置和训练参数，避免 500M 模型的 Adam 状态占用大量磁盘。确实需要保存优化器状态时再加：

```bash
--save_optimizer
```

## Generate

训练结束后用最终 checkpoint 做续写测试：

```bash
CUDA_VISIBLE_DEVICES=4 python generate.py \
  --checkpoint checkpoints/pretrain_500m/pretrain_step_XXXXX.pt \
  --tokenizer_path tokenizer \
  --prompt "人工智能是" \
  --max_new_tokens 120 \
  --temperature 0.8 \
  --top_k 50
```

这是预训练模型，不是指令助手。测试时更适合使用“续写式”提示，例如“人工智能是”“李白是唐代著名诗人，他”。

## Notes For M40

- 优先使用已经验证稳定的 GPU 组合，例如 `CUDA_VISIBLE_DEVICES=4,5`。
- 避开有 ECC double-bit 记录或残留异常进程的 GPU。
- 没有 `tmux` 时可以用 `nohup`，但推荐安装或使用 `tmux` 保存长任务会话。
- 长训练建议始终用 `tee` 保存日志，方便之后画 loss 曲线。
- FSDP 对通信比较敏感；如果 NCCL 或 timeout 问题频繁出现，优先退回 462M 配置并确认 GPU 组合稳定。
