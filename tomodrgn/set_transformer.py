"""
Set transformer classes adapted from https://github.com/juho-lee/set_transformer under MIT license
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    """
    Multihead Attention module.
    Performs attention function between two q-dimensional inputs.
    The first set matrix is treated as the Query matrix, while the second set matrix is treated as both the Key and Value matrices.
    Attention is multi-headed: input is projected to num_heads different matrices, attention is performed on each, and the concatenated outputs are fed through a linear activation.
    Overall dimensionality mapping is (input1: dim_q, input2: dim_q) -> dim_v, independent of the number of heads.
    """
    def __init__(self,
                 dim_q: int,
                 dim_k: int,
                 dim_v: int,
                 num_heads: int,
                 ln: bool = False):
        """
        Initialize a MAB instance.
        :param dim_q: dimensionality of input set
        :param dim_k: dimensionality of "key" matrix, should be the same as dim_q
        :param dim_v: dimensionality of "value" matrix
        :param num_heads: number of attention heads
        :param ln: whether to perform layer normalization
        """
        super().__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        if ln:
            self.ln0 = nn.LayerNorm(dim_v)
            self.ln1 = nn.LayerNorm(dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)

    def forward(self,
                matrix1: torch.Tensor,
                matrix2: torch.Tensor) -> torch.Tensor:
        mat_q = self.fc_q(matrix1)
        mat_k, mat_v = self.fc_k(matrix2), self.fc_v(matrix2)

        dim_split = self.dim_v // self.num_heads
        mat_q_split = torch.cat(mat_q.split(dim_split, 2), 0)
        mat_k_split = torch.cat(mat_k.split(dim_split, 2), 0)
        mat_v_split = torch.cat(mat_v.split(dim_split, 2), 0)

        mat_a = torch.softmax(mat_q_split.bmm(mat_k_split.transpose(1, 2)) / math.sqrt(self.dim_v), 2)
        mat_o = torch.cat((mat_q_split + mat_a.bmm(mat_v_split)).split(mat_q.size(0), 0), 2)
        mat_o = mat_o if getattr(self, 'ln0', None) is None else self.ln0(mat_o)
        mat_o = mat_o + F.relu(self.fc_o(mat_o))
        mat_o = mat_o if getattr(self, 'ln1', None) is None else self.ln1(mat_o)
        return mat_o


class SAB(nn.Module):
    """
    Set Attention module.
    Performs MultiHead Attention using the same input set as both inputs.
    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 num_heads: int,
                 ln: bool = False):
        super().__init__()
        self.mab = MAB(dim_q=dim_in,
                       dim_k=dim_in,
                       dim_v=dim_out,
                       num_heads=num_heads,
                       ln=ln)

    def forward(self, matrix1: torch.Tensor) -> torch.Tensor:
        return self.mab(matrix1, matrix1)


class PMA(nn.Module):
    """
    Pooled MultiHead Attention module.
    Performs MultiHead Attention between a set of seed vectors and a query matrix.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 num_seeds: int,
                 ln: bool = False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim_q=dim,
                       dim_k=dim,
                       dim_v=dim,
                       num_heads=num_heads,
                       ln=ln)

    def forward(self, matrix2: torch.Tensor) -> torch.Tensor:
        matrix1 = self.S.repeat(matrix2.size(0), 1, 1)
        return self.mab(matrix1, matrix2)


class SetTransformer(nn.Module):
    """
    Set Transformer module, the primary contribution of https://github.com/juho-lee/set_transformer.
    Consists of encoder and decoder modules.
    The encoder maps an input matrix of shape (num_set_elements, dim_input) to (num_set_elements, dim_hidden) embeddings using stacked Set Attention modules.
    The decoder aggregates all set embeddings using Pooled Multihead Attention, then models potential interactions between k seed-vector-derived outputs with Set Attention modules.
    Overall dimensionality mapping is (num_set_elements, dim_input) -> (num_outputs, dim_output).
    """
    def __init__(self,
                 dim_input: int,
                 num_outputs: int,
                 dim_output: int,
                 dim_hidden: int = 128,
                 num_heads: int = 4,
                 ln: bool = False):
        # dim_input is node count for layer 1 (should be out_dim_A, in paper this was 512 from VGG)
        # num_outputs is PMA num_seeds (k, should be 1?? or 2?? or large like 100??, can sweep)
        # dim_output is final dim of output (should be 2x zdim if directly used for VAE reparamaterization)
        # dim_hidden is node count for all hidden layers (can sweep)
        # num_heads is the number of heads for multihead attention in SAB (can maybe sweep, but gave errors in past)

        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=dim_input,
                dim_out=dim_hidden,
                num_heads=num_heads,
                ln=ln),
            SAB(dim_in=dim_hidden,
                dim_out=dim_hidden,
                num_heads=num_heads,
                ln=ln))
        self.dec = nn.Sequential(
            PMA(dim=dim_hidden,
                num_heads=num_heads,
                num_seeds=num_outputs,
                ln=ln),
            SAB(dim_in=dim_hidden,
                dim_out=dim_hidden,
                num_heads=num_heads,
                ln=ln),
            SAB(dim_in=dim_hidden,
                dim_out=dim_hidden,
                num_heads=num_heads,
                ln=ln),
            nn.Linear(in_features=dim_hidden,
                      out_features=dim_output))

    def forward(self, matrix):
        return self.dec(self.enc(matrix))
