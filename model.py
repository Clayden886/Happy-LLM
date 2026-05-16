from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    # tokenizer 词表大小，后面训练 tokenizer 时要和它保持一致
    vocab_size: int = 6144

    # 模型隐藏维度，也就是每个 token 会被表示成多长的向量
    dim: int = 512

    # Transformer Decoder 层数
    n_layers: int = 8

    # Query 注意力头数量
    n_heads: int = 8

    # Key / Value 注意力头数量
    # 如果为 None，就默认等于 n_heads，即普通多头注意力
    # 如果小于 n_heads，就是 GQA
    n_kv_heads: Optional[int] = None

    # 单条样本最大 token 长度
    max_seq_len: int = 512

    # dropout 概率
    dropout: float = 0.0

    # RMSNorm 的稳定项
    norm_eps: float = 1e-5

    # MLP 中间层维度的倍率
    multiple_of: int = 256

    # 是否使用 bias
    # LLaMA 类模型通常不用 bias
    bias: bool = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)

    freqs = torch.outer(t, freqs)

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim >= 2

    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]

    return freqs.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
):
    xq_float = xq.float()
    xk_float = xk.float()

    xq_real, xq_imag = xq_float[..., ::2], xq_float[..., 1::2]
    xk_real, xk_imag = xk_float[..., ::2], xk_float[..., 1::2]

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_real)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_real)

    xq_out_real = xq_real * freqs_cos - xq_imag * freqs_sin
    xq_out_imag = xq_real * freqs_sin + xq_imag * freqs_cos

    xk_out_real = xk_real * freqs_cos - xk_imag * freqs_sin
    xk_out_imag = xk_real * freqs_sin + xk_imag * freqs_cos

    xq_out = torch.stack([xq_out_real, xq_out_imag], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_real, xk_out_imag], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    x = x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    return x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout

        assert config.dim % config.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos[:seq_len], freqs_sin[:seq_len])

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim ** 0.5)

        mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=x.device,
        )
        mask = torch.triu(mask, diagonal=1)

        scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)

        output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=config.bias)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.dropout(x)

class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention = Attention(config)
        self.feed_forward = MLP(config)

        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
        )

        out = h + self.feed_forward(
            self.ffn_norm(h)
        )

        return out

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.output.weight = self.tok_embeddings.weight

        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = tokens.shape

        assert seq_len <= self.config.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
        )

        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seq_len].to(h.device)
        freqs_sin = self.freqs_sin[:seq_len].to(h.device)

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            logits = self.output(h[:, [-1], :])
            loss = None

        return logits, loss
    
    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None:
                if (idx_next == eos_token_id).all():
                    break

        return idx

