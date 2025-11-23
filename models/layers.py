from typing import Tuple
import einops
import torch
from torch import nn
import torch.nn.functional as F

#try:
#    from flash_attn_interface import flash_attn_func  # type: ignore[import]
#except ImportError:
#    # Fallback to FlashAttention 2
#    from flash_attn import flash_attn_func  # type: ignore[import]
from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)

class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class CastedConv1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, bias: bool,
        stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1
    ):
        super().__init__()
        # Truncated LeCun normal init
        weight = torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size
        )
        self.weight = nn.Parameter(
            trunc_normal_init_(weight, std=1.0 / (in_channels ** 0.5))
        )

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.conv1d(
            x,
            w,
            bias=b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class SSM(nn.Module):
    def __init__(
        self, hidden_size: int, inner_size: int,  kernel_size: int,
        state_size: int, dt_rank: int, conv_bias: bool = False, proj_bias: bool = False
    ):
        super().__init__()

        self.inner_size = inner_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.dt_rank = dt_rank

        self.in_proj = CastedLinear(hidden_size, inner_size * 2, bias=proj_bias)
        self.conv1d = CastedConv1d(
            in_channels=inner_size,
            out_channels=inner_size,
            bias=conv_bias,
            kernel_size=kernel_size,
            groups=inner_size,
            padding=kernel_size-1
        )

        self.x_proj = CastedLinear(inner_size, dt_rank + state_size * 2, bias=False)
        self.dt_proj = CastedLinear(dt_rank, inner_size, bias=True)

        A = einops.repeat(torch.arange(1, state_size+1), "n -> d n", d=inner_size)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(inner_size))
        self.out_proj = nn.Linear(inner_size, hidden_size, bias=proj_bias)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        (b, l, d) = hidden_states.shape
        hs_and_res = self.in_proj(hidden_states)
        (hidden_states, res) = hs_and_res.split(
            split_size=[self.inner_size, self.inner_size], dim=-1
        )

        hidden_states = einops.rearrange(hidden_states, "b l d_in -> b d_in l")
        hidden_states = self.conv1d(hidden_states)[:, :, :l]
        hidden_states = einops.rearrange(hidden_states, "b d_in l -> b l d_in")
        hidden_states = F.silu(hidden_states)

        y = self.ssm(hidden_states)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(
        self, x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
        B: torch.Tensor, C: torch.Tensor, D: torch.Tensor
    ) -> torch.Tensor:
        (b, l, d_in) = x.shape
        n = A.shape[1]

        delta_A = torch.exp(einops.einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        delta_Bx = einops.einsum(delta, B, x, " b l d_in, b l n, b l d_in -> b l d_in n")

        z = torch.zeros((b, d_in, n), device=delta_A.device)
        ys = []
        for i in range(l):
            z = delta_A[:, i] * z + delta_Bx[:, i]
            y = einops.einsum(z, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + x * D
        return y
