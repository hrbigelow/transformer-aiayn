from dataclasses import dataclass
import re
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
from torch.linalg import vector_norm
import torch.nn.functional as F
from .data import load_token_histo
from . import hparams
import pdb

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
        vval = t.einsum('bcm,hmv->bhcv', kvinput, self.wv)
        alogit = t.einsum('bhck,bhdk->bhcd', kval, qval)
        if self.masked:
            side = kvinput.shape[1]
            alogit += t.full_like(alogit, -100.0).triu()
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
        pos_embed = pos_embed.to(input.device)
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

class TeeSequential(nn.Module):
    """
    Like nn.Sequential, but accepts an additional global input in each
    module
    """
    def __init__(self, *mods):
        super().__init__()
        for l, mod in enumerate(mods):
            super().add_module(f'{l}', mod)
        # self.mods = mods

    def forward(self, side_input, input):
        out = input
        for mod in self.children():
            out = mod(side_input, out)
        return out

class Decoder(nn.Module):
    def __init__(self, hps, T, embed_matrix):
        super().__init__()
        self.embed_layer = InputEmbedding(embed_matrix, hps)
        self.body = TeeSequential(*(DecoderLayer(hps) for _ in
            range(hps.num_layers)))
        self.linear_final = nn.Linear(hps.M, T)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_out, dec_in):
        """
        enc_out: bcm 
        dec_in: bc
        returns: bct
        """
        out = self.embed_layer(dec_in)
        out = self.body(enc_out, out)
        out = self.linear_final(out)
        out = self.softmax(out)
        return out

class Model(nn.Module):
    def __init__(self, hps, token_info):
        super().__init__()
        self.rng = t.Generator()
        self.rng.manual_seed(hps.random_seed)
        token_histo = token_info['histo']
        self.pad_token_id = token_info['pad_token_id']
        T = token_histo.shape[0]
        self.embed_matrix = nn.Embedding(T,hps.M)
        self.encoder = Encoder(hps, self.embed_matrix)
        self.decoder = Decoder(hps, T, self.embed_matrix)
        self.loss = Loss(token_histo, self.pad_token_id) 

    def total_params(self):
        # get total number of parameters
        return sum(par.numel() for par in self.parameters())

    def param_shape_map(self):
        from collections import Counter
        shape_map = Counter(tuple(par.shape) for par in self.parameters())
        return

    def input_output_attention(self, enc_input, dec_input):
        """
        c, d are lengths in tokens
        enc_input: bc
        dec_input: bd
        returns: bcd, a matrix of attention probabilities

        Needed for beam search
        """
        pass

    def regex_params(self, pat):
        """
        For each parameter name matching pat, return:
        name => (*captures)
        """
        ret = {}
        for name, par in self.named_parameters():
            m = re.match(pat, name)
            if m:
                ret[name] = m.groups()
        return ret

    def get_ordered_params(self, pat):
        """
        Return an ordered list of parameters matching pat.
        Ordered according to any capture groups in pat.
        """
        params = []
        for name, par in self.named_parameters():
            m = re.match(pat, name)
            if m:
                params.append((m.groups(), name, par))
        return [ (name, par) for _, name, par in sorted(params) ]

    def zero_gradients(self, pat):
        """
        Zero gradients matching parameter name `pat`.
        Return a map of the recorded gradients to restore later

        Usage:

        current_grads = zero_gradients(pat)

        # compute gradients for given samples
        loss.backward()

        # add in gradients
        self.add_gradients(current_grads)
    
        """
        grads = {} 
        params = self.get_ordered_params(pat)
        for name, par in params:
            if par.grad is None:
                continue
            grads[name] = par.grad.clone()
            par.grad.zero_()
        return grads

    def add_gradients(self, grads):
        """
        Add previously saved gradients back.
        Call this function if you are inspecting some gradients within an
        accumulation loop
        """
        for name, grad in grads.items():
            par = self.get_parameter(name)
            with t.no_grad():
                par.grad += grad

    def grad_norms(self, pat, index_dims=None):
        """
        Return a vector of gradient norms for all parameters matching pat.
        Entries in the vector will be ordered by any tuple of match patterns
        """
        if index_dims is None:
            index_dims = tuple()

        params = self.get_ordered_params(pat)
        norms = []

        for name, par in params:
            dims = [d for d in range(par.grad.ndim) if d not in index_dims]
            norms.append(vector_norm(par.grad, dim=dims).item())
        return norms

    def get_state(self):
        return dict(weights=self.state_dict(), rng=self.rng.get_state())

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
        self.pad_value = pad_value
        token_histo = F.normalize(token_histo.to(t.float64), p=1.0, dim=0)
        self.register_buffer('u', token_histo, persistent=False)
        self.vocab_size = self.u.shape[0]

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
        total_targets = dec_mask.sum()
        # target_fraction = total_targets / dec_mask.numel()
        # print(f'{total_targets=}, {target_fraction=}')
        return masked.sum() / total_targets

class CustomScheduler:
    def __init__(self, optimizer, M, warmup_steps):
        self.M = M
        # from section 5.3, page 7, equation 3
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer

    def update(self, step): 
        self.step = step
        ord_step = step + 1
        factor = min(ord_step ** -0.5, ord_step * self.warmup_steps ** -1.5)
        new_lr = self.M ** -0.5 * factor
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

    def current_lr(self):
        return self.optimizer.param_groups[0]['lr']

