from typing import Callable, Tuple
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np

def SSFM(
        x: torch.Tensor,
        p: int,
) -> Tuple[Callable, Callable]:
    band = x.shape[1]
    p = min(p, band // 2)
    with torch.no_grad():
        x = x / x.norm(dim=-1, keepdim=True)
        x_1, x_2 = x[..., ::2, :], x[..., 1::2, :]
        sim = x_1 @ x_2.transpose(-1, -2)
        _, top_indices = sim.max(dim=-1)
        so_id = top_indices.argsort(dim=-1, descending=True)[..., None]
        sk_id = so_id[..., p:, :]
        se_id = so_id[..., :p, :]
        m = top_indices[..., None].gather(dim=-2, index=se_id)

    def ssfm_c(feature_data: torch.Tensor, operation="mean") -> torch.Tensor:
        part_a, part_b = feature_data[..., ::2, :], feature_data[..., 1::2, :]
        N, _, C = part_a.shape
        sk = part_a.gather(dim=-2, index=sk_id.expand(N, -1, C))
        se_a = part_a.gather(dim=-2, index=se_id.expand(N, -1, C))
        com_b = part_b.scatter_reduce(-2, m.expand(N, -1, C), se_a, reduce=operation)
        merged_parts = [sk, com_b]
        return torch.cat(merged_parts, dim=1)

    def ssfm_r(data: torch.Tensor) -> torch.Tensor:
        sk_len = sk_id.shape[1]
        sk, com = data.split([sk_len, data.shape[1] - sk_len], dim=-2)
        N, _, C = sk.shape
        res_a = com.gather(dim=-2, index=m.expand(-1, -1, C))
        output = torch.zeros_like(x).scatter_(-2, (2 * sk_id).expand(-1, -1, C), sk)
        output = output.scatter_(-2, (2 * se_id).expand(-1, -1, C), res_a)
        output[..., 1::2, :] = com
        return output

    return ssfm_c, ssfm_r



class GSAM(nn.Module):
    def __init__(self, n, num_patches, d=64, eps=0):
        super(GSAM, self).__init__()
        self.n = n
        self.num_patches = num_patches
        self.d = d
        self.gamma = nn.Parameter(torch.rand((2, d)))
        self.beta = nn.Parameter(torch.rand((2, d)))
        self.b = nn.Parameter(torch.rand((n, n)))
        self.w1 = nn.Linear(num_patches, 2 * num_patches + d)
        self.w2 = nn.Linear(num_patches, num_patches)
        self.LayerNorm = nn.LayerNorm(num_patches, eps=eps)
        self.g = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x0, x = x, self.LayerNorm(x)
        p, v, o = torch.split(self.g(self.w1(x)), [self.num_patches, self.num_patches, self.d], dim=-1)
        o = torch.einsum("...r, hr->...hr", o, self.gamma) + self.beta
        q, k = torch.unbind(o, dim=-2)
        qk = torch.einsum("bnd,bmd->bnm", q, self.drop(k))
        Z = torch.square(F.relu(qk / np.sqrt(self.d) + self.b))/self.n
        x = p * torch.einsum("bnm, bme->bne", Z, v)
        x = self.w2(x) + self.drop(x0)
        return x, Z

class GSTAM(nn.Module):
    def __init__(self, num_patches, d=64, eps=0,):
        super(GSTAM, self).__init__()
        self.num_patches = num_patches
        self.d = d
        self.w1 = nn.Linear(num_patches, 2 * num_patches + d)
        self.w2 = nn.Linear(num_patches, num_patches)
        self.LayerNorm = nn.LayerNorm(num_patches, eps=eps)
        self.g = nn.ReLU()
        self.drop = nn.Dropout(0.5)


    def forward(self, x, Z2):
        x0, x = x, self.LayerNorm(x)
        p, v, _ = torch.split(self.g(self.w1(x)), [self.num_patches, self.num_patches, self.d], dim=-1)
        x = p * torch.einsum("bnm, bme->bne", self.drop(Z2), v)
        x = self.w2(x) + self.drop(x0)
        return x
