from dataclasses import dataclass
import math

from einops import rearrange
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from flux.math import attention, rope


def expand_attn_mask(
    hidden_states: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Expand a mask so that the text mask is included.
    """
    bsz = attn_mask.shape[0]
    assert bsz == hidden_states.shape[0]
    residual_seq_len = hidden_states.shape[2] # Note: hidden_states is of shape (bsz, num_heads, seq_len, head_dim)
    mask_seq_len = attn_mask.shape[1]

    expanded_mask = torch.ones(bsz, residual_seq_len)
    start_index = residual_seq_len - mask_seq_len
    expanded_mask[:, start_index:] = attn_mask
    
    expanded_mask = expanded_mask
    expanded_mask = (expanded_mask > 0).bool().to(hidden_states.device)

    return expanded_mask


def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training=True, threshold=0.5
):
    if training :
        # ~Gumbel(0,1)`
        with torch.random.fork_rng(devices=[logits.device]):
            gumbels1 = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
            gumbels2 = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class TokenSelect(nn.Module):
    def __init__(self, dim_in, num_sub_layer, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Linear(dim_in, num_sub_layer, bias=bias)

        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        b, l = x.shape[:2]
        logits = self.mlp_head(x)
        
        
        token_select = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        
        return token_select, logits
    
    
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)
    

class TDRouter(torch.nn.Module):
    def __init__(self, dim: int, threshold: float = 0.5, tau: float = 5, is_hard: bool = True):
        """
        Initialize the TDRouter layer.

        Args:
            dim (int): The dimension of the input tensor.
            cond_dim (int): The dimension of the conditional tensor.
            threshold (float): The threshold for the router, determing the ratio of droped tokens.

        Attributes:
            weight (nn.Parameter): Learnable router parameter.

        """
        super().__init__()
        self.tau = tau
        self.is_hard = is_hard
        self.threshold = threshold
        self.fc = nn.Linear(
            dim,
            1,
            bias=True,
        )

    def forward(self, token, cond):
        """
        Forward pass through the TDRouter layer.

        Args:
            token (torch.Tensor): The input token tensor.
            cond (torch.Tensor): The conditional input tensor.

        Returns:
            indices (torch.Tensor): The output tensor after applying TDRouter.
            logits (torch.Tensor): The logits of the TDRouter.
        """
        
        if len(token.shape) == 4:
            B, H, L, D = token.shape
            token = token.permute(0, 2, 1, 3).view(B, L, -1) # (batch, token length, feature)
        
        token = token + cond.unsqueeze(1)  # (batch, 1, feature) + (batch, token length, feature)

        logits = self.fc(token).squeeze(-1) # (batch, token length)
        mask = _gumbel_sigmoid(logits, training=self.training, hard=self.is_hard, threshold=self.threshold, tau=self.tau)
        
        # _, indices = _softmax.topk(int((1 - self.threshold) * seq_len), dim=-1, largest=True, sorted=True)

        return mask.to(logits.dtype), logits


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, attn_token_select: bool = False, mlp_token_select: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        
        self.attn_token_select = None
        self.mlp_token_select = None
        if attn_token_select:
            self.attn_token_select = TDRouter(hidden_size)
        # if mlp_token_select:
        #     self.mlp_token_select = TDRouter(hidden_size)

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, img_mask: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        pe_k = pe
        if self.attn_token_select:
            B, H, L, D = img_k.shape
            
            sub_token_select, token_logits = self.attn_token_select(img_v, vec)
            img_k = img_k * sub_token_select.view(B, 1, L, 1)
            img_v = img_v * sub_token_select.view(B, 1, L, 1)
            # txt_len = txt.shape[1]
            # select_L = sub_token_select.shape[1]
            # batch_indices = torch.arange(B).unsqueeze(1).expand(B, select_L)
            # img_k = img_k.transpose(1, 2)[batch_indices, sub_token_select].transpose(1, 2)
            # img_v = img_v.transpose(1, 2)[batch_indices, sub_token_select].transpose(1, 2)
            # img_mask = img_mask[batch_indices, sub_token_select]
            # img_pe = pe.squeeze(1)[:, txt_len:]
            # img_pe = img_pe[batch_indices, sub_token_select].unsqueeze(1)
            # pe_k = torch.cat((pe[:, :, :txt_len], img_pe), dim=2)
            
        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        
        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        
        if img_mask is not None:
            attn_mask = expand_attn_mask(q, img_mask)
            
        with torch.cuda.device(q.device.index):
            attn = attention(q, k, v, pe_q=pe, pe_k=pe_k, attn_mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # apply mlp token select if it exists
        if self.mlp_token_select:
            sub_token_select, token_logits = self.mlp_token_select(img, vec)
            token_indices = sub_token_select.unsqueeze(-1).expand(-1, -1, img.shape[-1]) # Shape [B, N, D]
            select_tokens = torch.gather(img, 1, token_indices) # Shape [B, N, D]
            select_tokens = img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(select_tokens) + img_mod2.shift)
            updated_img = torch.zeros_like(img)
            updated_img.scatter_(1, token_indices, select_tokens)
            img = img + updated_img
            sub_token_select, token_logits = None, None
            # Get the indices of the selected tokens
            # sub_token_select, token_logits = self.mlp_token_select(img)
            # indices = sub_token_select.squeeze(-1).nonzero(as_tuple=False)  # Shape [N, 2]
            # batch_indices, token_indices = indices[:, 0], indices[:, 1]     # Shape [N]

            # Extract the selected tokens
            # selected_tokens = img[batch_indices, token_indices, :]          # Shape [N, D]

            # Get the corresponding scale, shift, and gate values
            # scale_values = img_mod2.scale[batch_indices, 0, :]              # Shape [N, D]
            # shift_values = img_mod2.shift[batch_indices, 0, :]              # Shape [N, D]
            # gate_values = img_mod2.gate[batch_indices, 0, :]                # Shape [N, D]

            # Update the original img tensor in-place
            # img[batch_indices, token_indices, :] += gate_values * self.img_mlp((1 + scale_values) * self.img_norm2(selected_tokens) + shift_values)             # Shape [B, L, D]
        else:
            sub_token_select, token_logits = None, None
            img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt, sub_token_select, token_logits


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        attn_token_select: bool = False,
        mlp_token_select: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)
        
        self.attn_token_select = None
        if attn_token_select:
            self.attn_token_select = TDRouter(hidden_size)
        
    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, img_mask: Tensor) -> Tensor:
        
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        
        if img_mask is not None:
            attn_mask = expand_attn_mask(q, img_mask)
        
        pe_k = pe
        if self.attn_token_select:
            B, H, L, D = k.shape
            txt_len = L - img_mask.shape[1]
            sub_token_select, token_logits = self.attn_token_select(v, vec)
            k = k * sub_token_select.view(B, 1, L, 1)
            v = v * sub_token_select.view(B, 1, L, 1)
            # select_L = sub_token_select.shape[1]
            # batch_indices = torch.arange(B).unsqueeze(1).expand(B, select_L)
            # k = k.transpose(1, 2)[batch_indices, sub_token_select].transpose(1, 2)
            # v = v.transpose(1, 2)[batch_indices, sub_token_select].transpose(1, 2)
            # attn_mask = attn_mask[batch_indices, sub_token_select]
            # pe_k = pe.squeeze(1)[batch_indices, sub_token_select].unsqueeze(1)
        
        # compute attention
        with torch.cuda.device(q.device.index):
            attn = attention(q, k, v, pe_q=pe, pe_k=pe_k, attn_mask=attn_mask)
        
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        
        # apply token select if it exists
        # if self.mlp_token_select:
            # sub_token_select, token_logits = self.joint_token_select(x)
        # else:
        sub_token_select, token_logits = None, None
        
        return x + mod.gate * output, sub_token_select, token_logits


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x