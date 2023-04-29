from dataclasses import dataclass
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pdb

@dataclass
class HyperParams:
    H: int = 8 # heads
    K: int = 64 # key (d_k in paper)
    V: int = 64 # value (d_v in paper)
    M: int = 512 # model (d_model in paper)
    F: int = 2048 # feed-forward dimension (d_ff in paper)
    num_layers: int = 6
    T: int = 0 # number of tokens
    warmup_steps: int = 4000

    # Section 5.4: Regularization (P_drop = 0.1)
    dropout_rate: float = 0.1

    # mixture coefficient for positional encoding
    pos_encoding_factor: float = 0.01

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-head attention from section 3.2.2
    """
    def __init__(self, hps, masked=False):
        super().__init__()
        mscale = np.sqrt(hps.M) ** -1
        vscale = np.sqrt(hps.V) ** -1
        self.wq = nn.Parameter(t.randn(hps.H,hps.M,hps.K) * mscale)
        self.wk = nn.Parameter(t.randn(hps.H,hps.M,hps.K) * mscale)
        self.wv = nn.Parameter(t.randn(hps.H,hps.M,hps.V) * mscale)
        self.wo = nn.Parameter(t.randn(hps.H,hps.V,hps.M) * vscale)
        self.scale_factor = np.sqrt(hps.K)
        self.M = hps.M
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
    def __init__(self, hps):
        super().__init__()
        mscale = np.sqrt(hps.M) ** -1
        fscale = np.sqrt(hps.F) ** -1
        self.w1 = nn.Parameter(t.randn(hps.M,hps.F) * mscale)
        self.b1 = nn.Parameter(t.zeros(hps.F))
        self.w2 = nn.Parameter(t.randn(hps.F,hps.M) * fscale)
        self.b2 = nn.Parameter(t.zeros(hps.M))
        self.M = hps.M

    def forward(self, input):
        """
        input: bcm
        """
        s = F.relu(t.einsum('bcm,mf->bcf', input, self.w1) + self.b1)
        out = t.einsum('bcf,fm->bcm', s, self.w2) + self.b2
        return out

class DropoutAddAndNorm(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.dropout = nn.Dropout(hps.dropout_rate)
        self.layer_norm = nn.LayerNorm(hps.M)

    def forward(self, residual, proximal):
        proximal = self.dropout(proximal)
        return self.layer_norm(residual + proximal)

class InputEmbedding(nn.Module):
    def __init__(self, embedding: t.nn.Embedding, hps):
        super().__init__()
        self.embedding = embedding # T,M
        self.T = self.embedding.num_embeddings
        self.M = self.embedding.embedding_dim
        self.scale_factor = np.sqrt(self.T) ** -1 # my experiment
        self.pos_factor = hps.pos_encoding_factor

    def positional_embedding(self, num_positions):
        pos = t.arange(num_positions)
        denom = 10000 ** t.linspace(0, 1, self.M)
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
        # embed = self.embedding(input) * self.scale_factor
        embed = self.embedding(input)
        return embed + pos_embed * self.pos_factor

class EncoderLayer(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.att = MultiHeadAttention(hps)
        self.norm1 = DropoutAddAndNorm(hps)
        self.ff = PositionwiseFF(hps)
        self.norm2 = DropoutAddAndNorm(hps)

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
    def __init__(self, hps, embed_matrix):
        super().__init__()
        self.embed_layer = InputEmbedding(embed_matrix, hps)
        mods = (EncoderLayer(hps) for _ in range(hps.num_layers))
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
    def __init__(self, hps):
        super().__init__()
        self.mask_att = MultiHeadAttention(hps, masked=True)
        self.norm1 = DropoutAddAndNorm(hps)
        self.att2 = MultiHeadAttention(hps)
        self.norm2 = DropoutAddAndNorm(hps)
        self.ff = PositionwiseFF(hps)
        self.norm3 = DropoutAddAndNorm(hps)

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
    def __init__(self, hps, embed_matrix):
        super().__init__()
        self.embed_layer = InputEmbedding(embed_matrix, hps)
        self.body = (DecoderLayer(hps) for _ in range(hps.num_layers))
        self.linear_final = nn.Linear(hps.M, hps.T)
        self.softmax = nn.Softmax(dim=2)

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
    def __init__(self, hps=HyperParams()):
        super().__init__()
        self.embed_matrix = nn.Embedding(hps.T,hps.M)
        self.encoder = Encoder(hps, self.embed_matrix)
        self.decoder = Decoder(hps, self.embed_matrix)

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
    def __init__(self, token_histo, pad_value, smoothing_eps=0.1):
        """
        page 8, Label Smoothing: using eps = 0.1
        """
        super().__init__()
        self.eps = smoothing_eps
        self.u = token_histo / token_histo.sum() 
        self.vocab_size = self.u.shape[0]
        self.pad_value = pad_value

    @staticmethod
    def kldiv(q, p, axis):
        # compute D[q(x) || p(x)] over the axis dimension
        terms = t.special.xlogy(q, q) - (q * p.log())
        return terms.sum(axis=axis)

    def label_smooth(self, labels):
        """
        labels: bc
        returns: bct
        """
        labels = F.one_hot(labels, self.vocab_size)
        smoothed = (1.0 - self.eps) * labels + self.eps * self.u
        return smoothed

    def forward(self, dec_input, dec_output):
        """
        dec_input: bc
        dec_output: bct
        """
        # bc
        labels = dec_input[:,1:]
        smoothed_labels = self.label_smooth(labels)
        dec_mask = t.ne(labels, self.pad_value).to(t.float64)

        # bct
        dec_pred = dec_output[:,:-1,:]
        kldiv = self.kldiv(smoothed_labels, dec_pred, 2)
        masked = kldiv * dec_mask
        return masked.mean()

class CustomScheduler:
    def __init__(self, optimizer, hps):
        # from section 5.3, page 7, equation 3
        self.optimizer = optimizer
        self.M = hps.M
        self.warmup_steps = hps.warmup_steps

    def update(self, step): 
        ord_step = step + 1
        factor = min(ord_step ** -0.5, ord_step * self.warmup_steps ** -1.5)
        new_lr = self.M ** -0.5 * factor
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

    def current_lr(self):
        return self.optimizer.param_groups[0]['lr']
