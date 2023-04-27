from dataclasses import dataclass
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pdb

@dataclass
class Dims:
    H: int = 8 # heads
    K: int = 64 # key (d_k in paper)
    V: int = 64 # value (d_v in paper)
    M: int = 512 # model (d_model in paper)
    F: int = 2048 # feed-forward dimension (d_ff in paper)
    num_layers: int = 6
    T: int = 0 # number of tokens

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-head attention from section 3.2.2
    """
    def __init__(self, dims, masked=False):
        super().__init__()
        self.wq = nn.Parameter(t.randn(dims.H,dims.M,dims.K))
        self.wk = nn.Parameter(t.randn(dims.H,dims.M,dims.K))
        self.wv = nn.Parameter(t.randn(dims.H,dims.M,dims.V))
        self.wo = nn.Parameter(t.randn(dims.H,dims.V,dims.M))
        self.scale_factor = np.sqrt(dims.K)
        self.M = dims.M
        self.masked = masked

    def forward(self, kvinput, qinput):
        """
        kvinput: bcm 
        qinput: bcm 
        output: bcm 
        """
        kval = t.einsum('bcm,hmk->bhck', kvinput, self.wk)
        qval = t.einsum('bcm,hmk->bhck', qinput, self.wq)
        vval = t.einsum('bcm,hmk->bhck', kvinput, self.wv)
        alogit = t.einsum('bhck,bhdk->bhcd', kval, qval)
        if self.masked:
            side = alogit.shape[2]
            alogit += t.full((side, side), -100.0).triu()
        att = t.softmax(alogit * self.scale_factor ** -1, dim=2)
        pre = t.einsum('bhcd,bhcv->bhdv', att, vval)
        out = t.einsum('bhcv,hvm->bcm', pre, self.wo)
        return out

class PositionwiseFF(nn.Module):
    """
    Implements equation 2 (section 3.3)
    """
    def __init__(self, M, F):
        super().__init__()
        self.w1 = nn.Parameter(t.randn(M,F))
        self.b1 = nn.Parameter(t.zeros(F))
        self.w2 = nn.Parameter(t.randn(F,M))
        self.b2 = nn.Parameter(t.zeros(M))
        self.M = M

    def forward(self, input):
        """
        input: bcm
        """
        s = F.relu(t.einsum('bcm,mf->bcf', input, self.w1) + self.b1)
        out = t.einsum('bcf,fm->bcm', s, self.w2) + self.b2
        return out

class AddAndNorm(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.layer_norm = nn.LayerNorm(M)

    def forward(self, residual, proximal):
        return self.layer_norm(residual + proximal)

class InputEmbedding(nn.Module):
    def __init__(self, embedding: t.nn.Embedding):
        super().__init__()
        self.embedding = embedding # T,M
        self.T = self.embedding.num_embeddings
        self.M = self.embedding.embedding_dim
        self.scale_factor = np.sqrt(self.T)

    def positional_embedding(self, num_positions):
        pos = t.arange(num_positions)
        exp = t.arange(self.M) / self.M * t.log2(t.tensor(10000))
        denom = t.exp2(exp)
        arg = pos.unsqueeze(-1) / denom.unsqueeze(0)
        # it shouldn't matter what order embeddings are placed but here I follow
        # the paper, and alternate sin with cos
        pos_emb = t.empty(num_positions,self.M)
        pos_emb[:,::2] = t.sin(arg[:,::2])
        pos_emb[:,1::2] = t.cos(arg[:,1::2])
        return pos_emb

    def forward(self, input):
        """
        input: bc (values: integer-encoded token ID)
        output: bcm
        """
        C = input.shape[1]
        pos_embed = self.positional_embedding(C)
        embed = self.embedding(input) * self.scale_factor
        return embed + pos_embed

class EncoderLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.att = MultiHeadAttention(dims)
        self.norm1 = AddAndNorm(dims.M)
        self.ff = PositionwiseFF(dims.M, dims.F)
        self.norm2 = AddAndNorm(dims.M)

    def forward(self, input):
        """
        input: bcm
        returns: bcm
        """
        att = self.att(input, input)
        norm = self.norm1(input, att)
        ff = self.ff(norm)
        out = self.norm2(norm, ff)
        return out

class Encoder(nn.Module):
    def __init__(self, dims, embed_matrix):
        super().__init__()
        self.embed_layer = InputEmbedding(embed_matrix)
        mods = (EncoderLayer(dims) for _ in range(dims.num_layers))
        self.body = nn.Sequential(*mods)

    def forward(self, input):
        """
        input: bc
        returns: bcm
        """
        out = self.embed_layer(input)
        out = self.body(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mask_att = MultiHeadAttention(dims, masked=True)
        self.norm1 = AddAndNorm(dims.M)
        self.att2 = MultiHeadAttention(dims)
        self.norm2 = AddAndNorm(dims.M)
        self.ff = PositionwiseFF(dims.M, dims.F)
        self.norm3 = AddAndNorm(dims.M)

    def forward(self, enc_out, input):
        """
        enc_out: bcm
        queries: bcm
        """
        att1 = self.mask_att(input, input)
        norm1 = self.norm1(input, att1)
        att2 = self.att2(enc_out, norm1)
        norm2 = self.norm2(norm1, att2)
        ff = self.ff(norm2)
        out = self.norm3(norm2, ff)
        return out

class Decoder(nn.Module):
    def __init__(self, dims, embed_matrix):
        super().__init__()
        self.embed_layer = InputEmbedding(embed_matrix)
        self.body = (DecoderLayer(dims) for _ in range(dims.num_layers))
        self.linear_final = nn.Linear(dims.M, dims.T)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc_out, dec_in):
        """
        enc_out: bcm 
        dec_in: bc
        returns: bct
        """
        out = self.embed_layer(dec_in)
        for mod in self.body:
            out = mod(enc_out, out)
        out = self.linear_final(out)
        out = self.softmax(out)
        return out

class Model(nn.Module):
    def __init__(self, dims=Dims()):
        super().__init__()
        self.embed_matrix = nn.Embedding(dims.T,dims.M)
        self.encoder = Encoder(dims, self.embed_matrix)
        self.decoder = Decoder(dims, self.embed_matrix)

    def forward(self, enc_input, dec_input):
        """
        enc_input: bc
        dec_input: bc
        returns: bct
        """
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(enc_output, dec_input)
        return dec_output


class Loss(nn.Module):
    def __init__(self, token_histo, smoothing_eps=0.1):
        """
        page 8, Label Smoothing: using eps = 0.1
        """
        super().__init__()
        self.eps = smoothing_eps
        self.u = token_histo / token_histo.sum() 
        self.vocab_size = self.u.shape[0]

    def forward(self, dec_input, dec_output):
        """
        dec_input: bc
        dec_output: bct
        """
        # bct
        dec_target = F.one_hot(dec_input[:,1:], self.vocab_size)
        smooth_target = (1.0 - self.eps) * dec_target + self.eps * self.u
        dec_pred = dec_output[:,:-1,:]
        shape = (-1, dec_pred.shape[2])
        xent = F.cross_entropy(dec_pred.reshape(*shape), smooth_target.reshape(*shape))
        return xent

