# Happy LLM

一个从零搭建的小型 LLaMA-like 语言模型项目，用于学习中文 tokenizer 训练、预训练、SFT 微调和文本生成的完整流程。

## Features

- Decoder-only Transformer
- RMSNorm
- RoPE 旋转位置编码
- Causal Self-Attention
- SwiGLU MLP
- 自回归 `generate`
- BPE tokenizer 训练
- 预训练脚本，支持 DDP 和断点恢复
- SFT 脚本，支持 DDP 和断点恢复
- Hugging Face 开源数据集转换脚本

## Project Structure

```text
.
├── model.py                    # LLaMA-like Transformer
├── dataset.py                  # PretrainDataset / SFTDataset
├── train_pretrain.py           # 预训练入口
├── train_sft.py                # SFT 微调入口
├── generate.py                 # 推理生成入口
├── requirements.txt            # 非 PyTorch 依赖
├── scripts/
│   ├── train_tokenizer.py      # 训练 BPE tokenizer
│   └── prepare_datasets.py     # 下载并转换开源数据集
├── data/
│   ├── raw/                    # 原始数据，本地生成，不提交
│   └── processed/              # 处理后数据，本地生成，不提交
├── tokenizer/                  # tokenizer 产物，本地生成，不提交
└── checkpoints/                # 模型权重，本地生成，不提交
```

## Environment

推荐 Python 3.10。

```bash
conda create -n happy-llm python=3.10 -y
conda activate happy-llm
```

Tesla M40 服务器建议使用 CUDA 11.8 对应的 PyTorch：

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

本项目面向 NVIDIA Tesla M40 24GB 这类老卡做了保守设计：

- 默认使用 FP32
- 不依赖 Flash Attention
- 不依赖 xFormers / bitsandbytes
- 使用 PyTorch 原生 attention

## Data

项目使用两类数据。

预训练数据格式：

```jsonl
{"text": "这里是一段中文文本。"}
```

SFT 数据格式：

```jsonl
{"messages": [{"role": "system", "content": "你是一个AI助手。"}, {"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你？"}]}
```

也兼容简单 SFT 字段：

```jsonl
{"instruction": "中国的首都是哪里？", "output": "中国的首都是北京。"}
{"human": "中国的首都是哪里？", "assistant": "中国的首都是北京。"}
```

## Prepare Open Datasets

安装依赖后，可以用脚本准备开源数据。

准备 `hfl/alpaca_zh_51k`：

```bash
python scripts/prepare_datasets.py sft-alpaca-zh \
  --out_path data/processed/sft_alpaca_zh_51k.jsonl
```

先抽 1000 条测试：

```bash
python scripts/prepare_datasets.py sft-alpaca-zh \
  --out_path data/processed/sft_alpaca_zh_1k.jsonl \
  --max_samples 1000
```

准备 `BAAI/CCI3-HQ` 子集前，需要在 Hugging Face 页面接受访问条件，并设置 token：

```bash
export HF_TOKEN=your_huggingface_token
```

抽取 1 万条预训练文本：

```bash
python scripts/prepare_datasets.py pretrain-cci3-hq \
  --out_path data/processed/pretrain_cci3_hq.jsonl \
  --max_docs 10000 \
  --min_chars 80
```

按大小抽取，例如约 1GB：

```bash
python scripts/prepare_datasets.py pretrain-cci3-hq \
  --out_path data/processed/pretrain_cci3_hq_1gb.jsonl \
  --max_bytes 1073741824 \
  --min_chars 80
```

## Train Tokenizer

```bash
python scripts/train_tokenizer.py \
  --data_dir data/processed \
  --out_dir tokenizer \
  --vocab_size 6144
```

如果 tokenizer 输出大量 `<unk>`，说明语料太小或覆盖不足，需要换更大的中文语料重新训练。

## Pretrain

CPU 或 Mac 本地只建议用小模型做 smoke test：

```bash
python train_pretrain.py \
  --data_path data/processed/pretrain_cci3_hq.jsonl \
  --tokenizer_path tokenizer \
  --out_dir checkpoints/pretrain \
  --epochs 1 \
  --batch_size 2 \
  --max_seq_len 64 \
  --dim 128 \
  --n_layers 2 \
  --n_heads 4
```

服务器正式训练可以使用默认 30M 级配置：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_pretrain.py \
  --data_path data/processed/pretrain_cci3_hq.jsonl \
  --tokenizer_path tokenizer \
  --out_dir checkpoints/pretrain \
  --epochs 3 \
  --batch_size 2 \
  --max_seq_len 512 \
  --dim 512 \
  --n_layers 8 \
  --n_heads 8 \
  --learning_rate 3e-4 \
  --log_interval 10 \
  --save_interval 1000
```

断点恢复：

```bash
python train_pretrain.py \
  --resume checkpoints/pretrain/pretrain_step_1000.pt \
  --data_path data/processed/pretrain_cci3_hq.jsonl \
  --tokenizer_path tokenizer \
  --out_dir checkpoints/pretrain
```

## SFT

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sft.py \
  --data_path data/processed/sft_alpaca_zh_51k.jsonl \
  --tokenizer_path tokenizer \
  --out_dir checkpoints/sft \
  --pretrained_checkpoint checkpoints/pretrain/pretrain_step_1000.pt \
  --epochs 3 \
  --batch_size 2 \
  --max_seq_len 512 \
  --dim 512 \
  --n_layers 8 \
  --n_heads 8 \
  --learning_rate 1e-4 \
  --log_interval 10 \
  --save_interval 1000
```

断点恢复：

```bash
python train_sft.py \
  --resume checkpoints/sft/sft_step_1000.pt \
  --data_path data/processed/sft_alpaca_zh_51k.jsonl \
  --tokenizer_path tokenizer \
  --out_dir checkpoints/sft
```

## Generate

预训练模型续写：

```bash
python generate.py \
  --checkpoint checkpoints/pretrain/pretrain_step_1000.pt \
  --mode pretrain \
  --prompt "大语言模型" \
  --max_new_tokens 64
```

SFT 模型对话：

```bash
python generate.py \
  --checkpoint checkpoints/sft/sft_step_1000.pt \
  --mode sft \
  --prompt "中国的首都是哪里？" \
  --max_new_tokens 64
```

## Multi-GPU Notes

目标服务器是 8 张 Tesla M40 24GB。根据拓扑，建议优先测试：

```text
1 卡
2 卡：GPU0,1
4 卡：GPU0,1,2,3
```

不要一开始直接使用 8 卡。M40 通常没有 NVLink，跨 NUMA 通信可能导致 DDP 同步开销较高。正式训练前用日志中的 `tokens/s` 比较不同卡数效率。

## Recommended Workflow

```text
1. 在服务器安装环境
2. 准备 CCI3-HQ 子集和 alpaca_zh_51k
3. 用处理后的数据重新训练 tokenizer
4. 单卡跑 smoke test
5. 测试 2 卡 / 4 卡 tokens/s
6. 正式预训练
7. 从预训练 checkpoint 做 SFT
8. 用 generate.py 测试模型输出
```

## License And Dataset Notice

本项目代码用于学习和研究。使用开源数据集前，请确认对应数据集的 license、访问条件和用途限制。

特别注意：

- `BAAI/CCI3-HQ` 可能需要接受访问条件。
- `hfl/alpaca_zh_51k` 是中文指令数据，使用前应检查其数据来源和许可说明。
- 不要将受限制数据或大体积训练产物直接提交到 GitHub。
