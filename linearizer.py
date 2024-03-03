import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
import math





def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, hidden_dim * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 16, heads = 4, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

       

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

       

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

       
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)







class LinformerBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model) 
        self.Linformer_unit = LinformerSelfAttention(d_model, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout=dropout)            
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.Linformer_unit(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out






class LinearizerGatingUnit(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len,dropout):
        super().__init__()
        self.proj = nn.Linear(d_model,d_model)     
        self.Linz = LinformerBlock(
			   d_model, d_ffn, seq_len,dropout
			)

	
       

    def forward(self, x):
        u, v = x, x 
        u = self.proj(u)  
        v = self.Linz(v)
        out = u * v
        return out


class LinearizerBlock(nn.Module):
    def __init__(self, d_model,d_ffn,seq_len,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.lgu = LinearizerGatingUnit(d_model,d_ffn,seq_len,dropout)
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.lgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out



class Linearizer(nn.Module):
    def __init__(self, d_model, d_ffn,seq_len, num_layers,dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[LinearizerBlock(d_model,d_ffn,seq_len,dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








