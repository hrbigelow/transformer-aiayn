from dataclasses import dataclass
import re
import numpy as np
from . import hparams
from . import funcs
import pdb
import jax
import jax.numpy as jnp
import haiku as hk

def prepare(hkmod, instance_args, *call_args):
    def _fn(*call_args):
        mod = hkmod(*instance_args)
        return mod(*call_args)
    return hk.transform(_fn)

class MultiHeadAttention(hk.Module):
    """
    Implements Multi-head attention from section 3.2.2
    """
    def __init__(self, hps, masked=True):
        super().__init__(name='att')
        self.H = hps.H
        self.M = hps.M
        self.K = hps.K
        self.V = hps.V
        self.masked = masked
        self.mscale = np.sqrt(self.M) ** -1
        self.vscale = np.sqrt(self.V) ** -1
        self.scale_factor = np.sqrt(self.K)

    def __call__(self, kvinput, qinput):
        """
        kvinput: bcm 
        qinput: bcm 
        output: bcm 
        """
        kqv_init = hk.initializers.RandomNormal(self.mscale, 0.0)
        o_init = hk.initializers.RandomNormal(self.vscale, 0.0)
        dtype = kvinput.dtype

        wq = hk.get_parameter('wq', [self.H,self.M,self.K], dtype, kqv_init)
        wk = hk.get_parameter('wk', [self.H,self.M,self.K], dtype, kqv_init)
        wv = hk.get_parameter('wv', [self.H,self.M,self.V], dtype, kqv_init)
        wo = hk.get_parameter('wo', [self.H,self.V,self.M], dtype, o_init)

        kval = jnp.einsum('bcm,hmk->bhck', kvinput, wk)
        qval = jnp.einsum('bcm,hmk->bhck', qinput, wq)
        vval = jnp.einsum('bcm,hmv->bhcv', kvinput, wv)
        alogit = jnp.einsum('bhck,bhdk->bhcd', kval, qval)
        if self.masked:
            # query (d) can only attend to keys (c) at or before it 
            # mask[i,j] = -100 if i > j else 0
            mask = jnp.tri(alogit.shape[3], k=-1) * -10000.0
            # print(f'MHA.__call__: {self.masked=}, {mask.shape=}, {alogit.shape=}')
            alogit += mask 
        att = jax.nn.softmax(alogit * self.scale_factor ** -1, axis=2)
        pre = jnp.einsum('bhcd,bhcv->bhdv', att, vval)
        out = jnp.einsum('bhcv,hvm->bcm', pre, wo)
        return out

class PositionwiseFF(hk.Module):
    """
    Implements equation 2 (section 3.3)
    """
    def __init__(self, hps):
        super().__init__(name='ff')
        self.mscale = np.sqrt(hps.M) ** -1
        self.fscale = np.sqrt(hps.F) ** -1
        self.M = hps.M
        self.F = hps.F

    def __call__(self, input):
        """
        input: bcm
        """
        dtype = input.dtype
        w1_init = hk.initializers.RandomNormal(self.mscale, 0.0)
        w2_init = hk.initializers.RandomNormal(self.fscale, 0.0)
        w1 = hk.get_parameter('w1', [self.M,self.F], dtype, w1_init)
        b1 = hk.get_parameter('b1', [self.F], dtype, jnp.zeros)
        w2 = hk.get_parameter('w2', [self.F,self.M], dtype, w2_init)
        b2 = hk.get_parameter('b2', [self.M], dtype, jnp.zeros)
        s = jax.nn.relu(jnp.einsum('bcm,mf->bcf', input, w1) + b1)
        out = jnp.einsum('bcf,fm->bcm', s, w2) + b2
        return out

class DropoutAddAndNorm(hk.Module):
    def __init__(self, hps, is_train):
        super().__init__(name='res')
        self.is_train = is_train
        self.rate = hps.dropout_rate
        self.layer_norm = hk.LayerNorm(1, create_scale=True, create_offset=True,
                name='lnorm')

    def __call__(self, residual, proximal):
        """
        residual: bdm
        proximal: bdm
        output: bdm
        """
        if self.is_train:
            proximal = hk.dropout(hk.next_rng_key(), self.rate, proximal)
        return self.layer_norm(residual + proximal)

class EmbedMatrix(hk.Module):
    def __init__(self, T, M):
        super().__init__(name='embed_matrix')
        self.T = T
        self.M = M

    def __call__(self):
        init = hk.initializers.RandomNormal(1.0, 0.0)
        return hk.get_parameter('emb', [self.T, self.M], np.float32, init) 

class InputEmbedding(hk.Module):
    def __init__(self, embed_mat, hps):
        super().__init__(name='emb')
        self.embed_mat = embed_mat
        self.pos_factor = hps.pos_encoding_factor

    def positional_embedding(self, num_positions):
        pos = jnp.arange(num_positions)
        denom = 10000 ** jnp.linspace(0, 1, self.embed_mat.M)
        arg = jnp.expand_dims(pos, 1) / jnp.expand_dims(denom, 0)
        # it shouldn't matter what order embeddings are placed but here I follow
        # the paper, and alternate sin with cos
        pos_emb = jnp.empty((num_positions,self.embed_mat.M), np.float32)
        pos_emb = pos_emb.at[:,::2].set(jnp.sin(arg[:,::2]))
        pos_emb = pos_emb.at[:,1::2].set(jnp.cos(arg[:,1::2]))
        # C, M
        return pos_emb

    def __call__(self, input):
        """
        input: bc (values: integer-encoded token ID)
        output: bcm
        """
        C = input.shape[1]
        pos_embed = self.positional_embedding(C)
        # scaled_emb_mat = self.embed_mat() * self.scale_factor 
        embed = jnp.take(self.embed_mat(), input, axis=0)
        return embed + pos_embed * self.pos_factor

class EncoderLayer(hk.Module):
    def __init__(self, hps, is_train, layer_num):
        super().__init__(name=f'layer{layer_num:02d}')
        self.att = MultiHeadAttention(hps, masked=False)
        self.norm1 = DropoutAddAndNorm(hps, is_train)
        self.ff = PositionwiseFF(hps)
        self.norm2 = DropoutAddAndNorm(hps, is_train)

    def __call__(self, input):
        """
        input: bcm
        returns: bcm
        """
        att = self.att(input, input)
        norm = self.norm1(input, att)
        ff = self.ff(norm)
        out = self.norm2(norm, ff)
        return out

class Encoder(hk.Module):
    def __init__(self, hps, is_train, embed_layer):
        super().__init__(name='enc')
        self.embed_layer = embed_layer 
        self.layers = [EncoderLayer(hps, is_train, i) for i in range(hps.num_layers)]

    def __call__(self, input):
        """
        input: bc
        returns: bcm
        """
        out = self.embed_layer(input)
        for mod in self.layers:
            out = mod(out)
        return out

class DecoderLayer(hk.Module):
    def __init__(self, hps, is_train, layer_num):
        super().__init__(name=f'layer{layer_num:02d}')
        self.mask_att = MultiHeadAttention(hps, masked=True)
        self.norm1 = DropoutAddAndNorm(hps, is_train)
        self.att2 = MultiHeadAttention(hps, masked=False)
        self.norm2 = DropoutAddAndNorm(hps, is_train)
        self.ff = PositionwiseFF(hps)
        self.norm3 = DropoutAddAndNorm(hps, is_train)

    def __call__(self, enc_out, input):
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

class TeeSequential(hk.Module):
    """
    Like nn.Sequential, but accepts an additional global input in each
    module.  This is used for the Decoder, with the side-input being the Encoder
    output representation.
    """
    def __init__(self, *mods):
        super().__init__()
        self.layers = mods

    def __call__(self, side_input, input):
        out = input
        for mod in self.layers:
            out = mod(side_input, out)
        return out

class Decoder(hk.Module):
    def __init__(self, hps, is_train, T, embed_layer):
        super().__init__(name='dec')
        self.is_train = is_train
        self.embed_layer = embed_layer 
        self.body = TeeSequential(*(DecoderLayer(hps, is_train, i) for i in range(hps.num_layers)))
        # self.linear_final = hk.Linear(T)
        self.scale_factor = np.sqrt(hps.M) ** -1

    def __call__(self, enc_out, dec_in):
        """
        enc_out: bcm 
        dec_in: bc
        returns: bct
        """
        out = self.embed_layer(dec_in)
        # print(f'Decoder.__call__: {enc_out.shape=}, {out.shape=}')
        out = self.body(enc_out, out)
        scaled_emb_mat = self.embed_layer.embed_mat() * self.scale_factor
        # out = self.linear_final(out)
        out = jnp.einsum('bcm,tm -> bct', out, scaled_emb_mat)
        # out = jax.nn.softmax(out, axis=2)
        return out

class Model(hk.Module):
    def __init__(self, hps, is_train, token_histo, pad_token_id):
        super().__init__(name='tx')
        self.is_train = is_train
        self.hps = hps
        self.pad_token_id = pad_token_id 
        self.T = token_histo.shape[0]
        emb_mat = EmbedMatrix(self.T, hps.M) 
        self.embed_layer = InputEmbedding(emb_mat, hps) 
        self.encoder = Encoder(hps, is_train, self.embed_layer)
        self.decoder = Decoder(hps, is_train, self.T, self.embed_layer)

    def __call__(self, enc_input, dec_input, temperature=1.0):
        """
        enc_input: bc
        dec_input: bc
        returns: bct
        """
        if self.is_train:
            enc_output = self.encoder(enc_input)
            dec_output = self.decoder(enc_output, dec_input)
            return dec_output

        B = enc_input.shape[0]
        C = self.hps.max_sentence_length

        enc_output = self.encoder(enc_input)
        def sample_fn(i, dec_input): 
            rng_key = hk.next_rng_key()
            # print(f'{p=}, {enc_output.shape=}, {dec_input.shape=}')
            dec_output = self.decoder(enc_output, dec_input) / temperature
            dec_step = jax.lax.dynamic_slice(dec_output, (0, i, 0), (B, 1, self.T))
            sample = jax.random.categorical(rng_key, dec_step, axis=2)
            # print(f'{dec_step.shape=}, {sample.shape=}, {dec_input.shape=}')
            dec_input = dec_input.at[:,i+1].set(sample[:,0])
            return dec_input
        dec_input = jax.lax.fori_loop(0, C-1, sample_fn, dec_input)
        return dec_input

    def total_params(self):
        # get total number of parameters
        return sum(par.size for par in self.params_dict().values())

    def param_shape_map(self):
        from collections import Counter
        shape_map = Counter(tuple(par.shape) for par in self.params_dict().values())
        return

    def dec_enc_attention(self, enc_input, dec_input):
        """
        c, d are lengths in tokens
        enc_input: bc
        dec_input: bd
        returns: bcd, a matrix of attention probabilities

        Needed for beam search
        """
        # I will assume the attention used should be over the last layer
        pass


def predict(self, enc_input):
    alpha = self.hps.beam_search_alpha
    beta = self.hps.beam_search_beta
    beam_size = self.hps.beam_size
    max_length = self.hps.beam_search_maxlen 
    seq, score = funcs.beam_search(self, alpha, beta, beam_size, max_length, enc_input)
    return seq

class Objective(hk.Module):
    def __init__(self, token_histo, pad_value, smoothing_eps=0.1):
        """
        page 8, Label Smoothing: using eps = 0.1
        """
        super().__init__()
        # self.eps = smoothing_eps
        self.pad_value = pad_value
        token_histo = token_histo.astype(jnp.float32)
        self.u = token_histo / jnp.sum(token_histo)
        self.eps = smoothing_eps
        # self.register_buffer('u', token_histo, persistent=False)
        self.T = token_histo.shape[0] 

    def label_smooth(self, labels):
        """
        labels: bc
        returns: bct
        """
        one_hot = jax.nn.one_hot(labels, self.T, axis=-1)
        smoothed = (1.0 - self.eps) * one_hot + self.eps * self.u
        return smoothed

    @staticmethod
    def masked_mean(values, mask):
        # compute the mean of values at mask
        masked_values = values * mask
        return masked_values.sum() / mask.sum()

    def __call__(self, dec_input, dec_output_logits):
        """
        dec_input: bc
        dec_output: bct
        """
        # bc
        labels = dec_input[:,1:]
        smoothed_labels = self.label_smooth(labels)
        dec_mask = jnp.not_equal(labels, self.pad_value).astype(jnp.float32)
        # jax.debug.print('{}', dec_mask)
        dec_pred_logits = dec_output_logits[:,:-1,:]
        dec_pred = jax.nn.softmax(dec_pred_logits, axis=2)
        kldiv = funcs.fused_kldiv_softmax(smoothed_labels, dec_pred_logits, 2)
        # print(f'{smoothed_labels.shape=}, {dec_pred_logits.shape=}, {kldiv.shape=}, {dec_mask.shape=}')
        mean_kldiv = self.masked_mean(kldiv, dec_mask)
        label_entropy = self.masked_mean(funcs.entropy(smoothed_labels, axis=2), dec_mask)
        model_entropy = self.masked_mean(funcs.entropy(dec_pred, axis=2), dec_mask)
        # TODO: don't return model_entropy, it is wrong
        return mean_kldiv, label_entropy, model_entropy

def _wrap_haiku(mod_cls, *args):
    def wrapped_fn(*call_args):
        mod = mod_cls(*args)
        return mod(*call_args)
    return wrapped_fn

def make_model(hps, is_train, token_info):
    return hk.transform(_wrap_haiku(Model, hps, is_train, token_info['histo'],
        token_info['pad_token_id']))

def make_objective(hps, token_info):
    return hk.transform(_wrap_haiku(Objective, token_info['histo'],
        token_info['pad_token_id'], hps.label_smooth_eps))

