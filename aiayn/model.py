import functools
from dataclasses import dataclass
import re
import numpy as np
from . import hparams
from . import funcs
import pdb
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk

class MultiHeadAttention(hk.Module):
    """
    Implements Multi-head attention from section 3.2.2
    """
    def __init__(self, is_self_attn, H, M, K, V):
        super().__init__(name='att')
        self.H = H
        self.M = M
        self.K = K
        self.V = V
        self.mscale = np.sqrt(self.M) ** -1
        self.vscale = np.sqrt(self.V) ** -1
        self.scale_factor = np.sqrt(self.K)
        self.self_attn = is_self_attn

    def __call__(self, qinput, kvinput, qmask, tmask, qtmask):
        """
        all masks have 0 or 1.  1 means ignore (mask out)
        qinput: bqm 
        kvinput: btm 
        qmask: bq  (ignore these query positions)
        tmask: bt  (ignore these target positions)
        qtmask: bqt  (ignore combinations of query, target positions)
        output: bqm 
        """
        B,Q,_ = qinput.shape
        _,T,_ = kvinput.shape

        if qmask is None:
            qmask = jnp.zeros((B,Q))
        if tmask is None:
            tmask = jnp.zeros((B,T))
        if qtmask is None:
            qtmask = jnp.zeros((B,Q,T))

        qmask = qmask[:,:,None]
        tmask = tmask[:,None,:]
        logits_mask = jnp.maximum(tmask, qtmask)
        # jax.debug.print('tmask[0]:\n{}', tmask[0])
        # jax.debug.print('qtmask[0]:\n{}', qtmask[0])
        # jax.debug.print('logits_mask[0]:\n{}', logits_mask[0])

        kqv_init = hk.initializers.RandomNormal(self.mscale, 0.0)
        o_init = hk.initializers.RandomNormal(self.vscale, 0.0)
        dtype = kvinput.dtype

        wq = hk.get_parameter('wq', [self.H,self.M,self.K], dtype, kqv_init)
        wk = hk.get_parameter('wk', [self.H,self.M,self.K], dtype, kqv_init)
        wv = hk.get_parameter('wv', [self.H,self.M,self.V], dtype, kqv_init)
        wo = hk.get_parameter('wo', [self.H,self.V,self.M], dtype, o_init)

        query = jnp.einsum('hmd,bqm->bhqd', wq, qinput)
        key = jnp.einsum('hmd,btm->bhtd', wk, kvinput)
        val = jnp.einsum('hmd,btm->bhtd', wv, kvinput)
        
        logit_adj = jnp.expand_dims(logits_mask, 1) * -1e6
        alogit = jnp.einsum('bhqd,bhtd->bhqt', query, key) + logit_adj
        att = jax.nn.softmax(alogit * self.scale_factor ** -1, axis=3)
        pre = jnp.einsum('bhqt,bhtd->bhqd', att, val)
        out = jnp.einsum('hdm,bhqd->bqm', wo, pre)
        out = out * (1.0 - qmask) # to protect gradients of masked positions
        # jax.debug.print('qinput: {}\nkvinput: {}\n', qinput, kvinput)
        return out

    def get_keys_values(self, kvinput):
        """
        kvinput: btm
        returns: bhstd, the kvcache for this layer
        """
        kqv_init = hk.initializers.RandomNormal(self.mscale, 0.0)
        o_init = hk.initializers.RandomNormal(self.vscale, 0.0)
        dtype = kvinput.dtype

        assert self.K == self.V
        wk = hk.get_parameter('wk', [self.H,self.M,self.K], dtype, kqv_init)
        wv = hk.get_parameter('wv', [self.H,self.M,self.V], dtype, kqv_init)
        wkv = jnp.concatenate((wk[:,:,None,:], wv[:,:,None,:]), 2)

        return jnp.einsum('hmsd,btm->bhstd', wkv, kvinput)

    def incremental(self, layer, step, kvcache, new_toks):
        """
        kvcache: lbhstd (s=0 means key, s=1 means val) (from encoder or decoder)
        new_toks: b1m
        i: index into kcache and vcache where new_toks resides

        returns a tuple of:
        kvcache (updated at position i)
        out: b1m 
        """
        assert isinstance(layer, int), f'{layer=}'
        assert kvcache.ndim == 6, f'{kvcache.shape=}'
        assert new_toks.ndim == 3, f'{new_toks.shape=}'
        new_toks = new_toks[:,0,:]

        kqv_init = hk.initializers.RandomNormal(self.mscale, 0.0)
        o_init = hk.initializers.RandomNormal(self.vscale, 0.0)
        dtype = new_toks.dtype

        wq = hk.get_parameter('wq', [self.H,self.M,self.K], dtype, kqv_init)
        wk = hk.get_parameter('wk', [self.H,self.M,self.K], dtype, kqv_init)
        wv = hk.get_parameter('wv', [self.H,self.M,self.V], dtype, kqv_init)
        wo = hk.get_parameter('wo', [self.H,self.V,self.M], dtype, o_init)

        query = jnp.einsum('hmd,bm->bhd', wq, new_toks)
        wkv = jnp.concatenate((wk[:,:,None,:], wv[:,:,None,:]), 2)
        kv_next = jnp.einsum('hmsd,bm->bhsd', wkv, new_toks)

        if self.self_attn:
            kv_next_unsq = kv_next[None,:,:,:,None,:]
            kvcache = jax.lax.dynamic_update_slice(kvcache, kv_next_unsq, (layer,0,0,0,step,0))
            C = kvcache.shape[4]
            mask = jnp.greater(jnp.arange(C), step).astype(jnp.int32)

        kcache_layer = kvcache[layer,:,:,0,:,:]
        attn_logit = jnp.einsum('bhd,bhtd->bht', query, kcache_layer)
        # if not self.self_attn:
            # jax.debug.print('layer: {}, kcache[0,0,:,0:5]: {}', layer,
                    # kcache_layer[0,0,:,0:5])

        if self.self_attn:
            attn_logit = attn_logit + mask * -1e6
            # jax.debug.print('step {}, logit: {}', step, logit[0])

        attn_coeff = jax.nn.softmax(attn_logit * self.scale_factor ** -1, axis=2)
        pre = jnp.einsum('bht,bhtd->bhd', attn_coeff, kvcache[layer,:,:,1])
        out = jnp.einsum('hdm,bhd->bm', wo, pre)

        coeff_summary = attn_coeff.sum(axis=1) # sum over heads

        if self.self_attn: 
            return kvcache, out[:,None,:]
        else:
            return coeff_summary, out[:,None,:]

class PositionwiseFF(hk.Module):
    """
    Implements equation 2 (section 3.3)
    """
    def __init__(self, M, F):
        super().__init__(name='ff')
        self.mscale = np.sqrt(M) ** -1
        self.fscale = np.sqrt(F) ** -1
        self.M = M
        self.F = F

    def __call__(self, input, qmask=None):
        """
        input: bqm
        qmask: bq
        returns: bqm
        """
        if qmask is None:
            B,Q,_ = input.shape
            qmask = jnp.zeros((B,Q))

        dtype = input.dtype
        w1_init = hk.initializers.RandomNormal(self.mscale, 0.0)
        w2_init = hk.initializers.RandomNormal(self.fscale, 0.0)
        w1 = hk.get_parameter('w1', [self.M,self.F], dtype, w1_init)
        b1 = hk.get_parameter('b1', [self.F], dtype, jnp.zeros)
        w2 = hk.get_parameter('w2', [self.F,self.M], dtype, w2_init)
        b2 = hk.get_parameter('b2', [self.M], dtype, jnp.zeros)
        s = jax.nn.relu(jnp.einsum('mf, bqm -> bqf', w1, input) + b1)
        out = jnp.einsum('fm, bqf -> bqm', w2, s) + b2
        out = out * (1.0 - jnp.expand_dims(qmask, 2))
        # jax.debug.print('ff_out: {}', out)
        return out

class DropoutAddAndNorm(hk.Module):
    def __init__(self, dropout_rate, is_train, layer_num):
        super().__init__(name='res')
        self.layer_num = layer_num
        self.is_train = is_train
        self.rate = dropout_rate

        # TODO: What is the meaning of the 'axis' argument here?
        # If I choose 1, then the grad_tests for decoder_layer fails, causing
        # leakage across autoregressive positions
        # Tim says: only apply across embedding axis, not batch axis
        self.layer_norm = hk.LayerNorm(axis=(2,), create_scale=True, create_offset=True,
                name='lnorm')

    def __call__(self, residual, proximal, qmask=None):
        """
        residual: bdm
        proximal: bdm
        qmask: bd
        output: bdm
        """
        if qmask is None:
            B,D,_ = residual.shape
            qmask = jnp.zeros((B,D))

        assert residual.ndim == 3, f'{residual.ndim=}'
        assert proximal.ndim == 3, f'{proximal.ndim=}'
        assert qmask.ndim == 2

        if self.is_train:
            proximal = hk.dropout(hk.next_rng_key(), self.rate, proximal)
        add = (residual + proximal) * (1.0 - qmask[:,:,None])
        jax.debug.print('{} residual norm:\n{}',
                self.module_name,
                jnp.sqrt(jnp.power(residual, 2).sum(axis=2)))
        jax.debug.print('{} proximal norm:\n{}',
                self.module_name,
                jnp.sqrt(jnp.power(proximal, 2).sum(axis=2)))

        # return add
        return self.layer_norm(add)

class EmbedMatrix(hk.Module):
    def __init__(self, V, M):
        super().__init__(name='embed_matrix')
        self.V = V
        self.M = M

    def __call__(self):
        init = hk.initializers.RandomNormal(1.0, 0.0)
        return hk.get_parameter('emb', [self.V, self.M], np.float32, init) 

class InputEmbedding(hk.Module):
    def __init__(self, embed_mat, pos_factor):
        super().__init__(name='emb')
        self.embed_mat = embed_mat
        self.pos_factor = pos_factor 

    def positional_embedding(self, seq_positions):
        # seq_positions: bc
        denom = 10000 ** jnp.linspace(0, 1, self.embed_mat.M)
        arg = seq_positions[:,:,None] / denom[None,None,:]
        # it shouldn't matter what order embeddings are placed but here I follow
        # the paper, and alternate sin with cos
        pos_emb = jnp.empty_like(arg)
        pos_emb = pos_emb.at[:,::2].set(jnp.sin(arg[:,::2]))
        pos_emb = pos_emb.at[:,1::2].set(jnp.cos(arg[:,1::2]))
        return pos_emb

    def __call__(self, seq_tokens, seq_positions):
        """
        seq_tokens: bc (end-to-end packed tokens from sentences)
        seq_positions: bc (ids identifying position of token in a sentence)
        output: bcm
        """
        pos_embed = self.positional_embedding(seq_positions)
        # jax.debug.print('pos_factor: {}, pos_embed[0]:\n{}', self.pos_factor, pos_embed[0])
        # jax.debug.print('seq_positions: {}', seq_positions[0])
        embed = jnp.take(self.embed_mat(), seq_tokens, axis=0)
        
        # embed = funcs.take(self.embed_mat(), input.astype(jnp.float32), axis=0)
        # jax.debug.print('embed: {}\npos_embed: {}\n', embed, pos_embed)
        return embed + pos_embed * self.pos_factor

class EncoderLayer(hk.Module):
    def __init__(self, dropout_rate, arch, is_train, layer_num):
        super().__init__(name=f'layer{layer_num:02d}')
        H, M, K, V, F = tuple(arch[l] for l in 'HMKVF') 
        self.layer_num = layer_num
        self.is_train = is_train
        self.dropout_rate = dropout_rate
        self.att = MultiHeadAttention(True, H, M, K, V)
        self.norm1 = hk.LayerNorm((2,), True, True, name='lnorm1')
        self.norm2 = hk.LayerNorm((2,), True, True, name='lnorm2')
        self.ff = PositionwiseFF(M, F)

    def __call__(self, input, position_mask, qt_mask):
        """
        input: btm
        position_mask: bt
        qt_mask: bqt
        returns: bqm

        Architecture is the pre-LN Layer as described in https://arxiv.org/pdf/2002.04745.pdf
        but with additional dropout added in the same position as in https://arxiv.org/abs/1706.03762

        In my experiments, post-LN resulted in homogenization of the encoder
        embeddings (all embedding vectors at every position were nearly identical)
        """
        norm1 = self.norm1(input)
        att = self.att(norm1, norm1, position_mask, position_mask, qt_mask)
        if self.is_train:
            att = hk.dropout(hk.next_rng_key(), self.dropout_rate, att)
        post_add1 = input + att
        norm2 = self.norm2(post_add1)
        ff = self.ff(norm2, position_mask)
        if self.is_train:
            ff = hk.dropout(hk.next_rng_key(), self.dropout_rate, ff)
        out = post_add1 + ff 
        # jax.debug.print('encoder_layer {}: out[0,:,100:110]:\n{}', 
                # self.layer_num, out[0,:,100:110])
        return out 

class Encoder(hk.Module):
    def __init__(self, dropout_rate, arch, is_train, pos_enc_factor, embed_mat):
        super().__init__(name='enc')
        self.embed_layer = InputEmbedding(embed_mat, pos_enc_factor) 
        self.layers = [EncoderLayer(dropout_rate, arch, is_train, i) for i in range(arch['L'])]

    def __call__(self, seqs, seqids, tokids):
        """
        seqs: bt    (tokens)
        seqids: bt  (ids grouping tokens into separate sequences)
        tokids: bt  (-1 is non-sample, >=0 are token ids, i.e. positions within a sentence)
        returns: bqm
        """
        position_mask = jnp.equal(tokids, -1).astype(jnp.int32)
        qt_mask = jnp.not_equal(seqids[:,None,:], seqids[:,:,None]).astype(jnp.int32)

        input_embed = self.embed_layer(seqs, tokids) 
        out = input_embed
        # pdb.set_trace()
        # jax.debug.print('out: {}', out)
        for mod in self.layers:
            out = mod(out, position_mask, qt_mask)
        # jax.debug.print('encoder_out: {}', out)
        return out

    def from_embedding(self, embed, seqids):
        """
        embed: btm  (input embedding)
        seqids: bt  (ids grouping tokens into separate sequences, -1 for non-sample)
        """
        position_mask = jnp.equal(seqids, -1).astype(jnp.int32)
        qt_mask = jnp.not_equal(seqids[:,None,:], seqids[:,:,None]).astype(jnp.int32)

        out = embed
        for mod in self.layers:
            out = mod(out, position_mask, qt_mask)
        return out


class DecoderLayer(hk.Module):
    def __init__(self, dropout_rate, arch, is_train, layer_num):
        super().__init__(name=f'layer{layer_num:02d}')
        H, M, K, V, F = tuple(arch[l] for l in 'HMKVF') 
        self.dropout_rate = dropout_rate
        self.is_train = is_train
        self.self_att = MultiHeadAttention(True, H, M, K, V)
        self.cross_att = MultiHeadAttention(False, H, M, K, V)
        self.ff = PositionwiseFF(M, F)
        self.norm1 = hk.LayerNorm((2,), True, True, name='lnorm1')
        self.norm2 = hk.LayerNorm((2,), True, True, name='lnorm2')
        self.norm3 = hk.LayerNorm((2,), True, True, name='lnorm3')

    def __call__(self, enc_out, input, position_mask, qt_self_mask, qt_cross_mask):
        """
        all masks have 0 or 1 (1 = mask out)
        enc_out: btm (t from encoder)
        input: bqm 
        position_mask: bq 
        qt_self_mask: bqt
        qt_cross_mask: bqt
        out: bqm
        """
        B, Cdec, _ = input.shape

        # here both q and t are indexing into the decoder only
        # qt_causal_mask = 1.0 - jnp.tri(Cdec, k=0)
        # qt_self_mask = jnp.maximum(qt_causal_mask, qt_self_mask)

        norm1 = self.norm1(input) 
        att1 = self.self_att(norm1, norm1, position_mask, position_mask, qt_self_mask)
        if self.is_train:
            att1 = hk.dropout(hk.next_rng_key(), self.dropout_rate, att1)
        post_add1 = input + att1
        norm2 = self.norm2(post_add1)

        att2 = self.cross_att(norm2, enc_out, position_mask, None, qt_cross_mask)
        if self.is_train:
            att2 = hk.dropout(hk.next_rng_key(), self.dropout_rate, att2)
        post_add2 = post_add1 + att2
        norm3 = self.norm3(post_add2)
        ff = self.ff(norm3, position_mask)
        if self.is_train:
            ff = hk.dropout(hk.next_rng_key(), self.dropout_rate, ff)
        out = post_add2 + ff
        return out

    def enc_kvcache(self, enc_out):
        # return bhstd  (s in [0, 1], key or val)
        return self.cross_att.get_keys_values(enc_out)

    def incremental(self, layer, step, enc_kvcache, dec_kvcache, xattn, next_embed):
        """
        enc_kvcache: lbhstd 
        dec_kvcache: lbhsqd
        xattn:  bt  (cumulative attention on encoder embeddings)
        next_embed: b1m
        """
        norm1 = self.norm1(next_embed)
        dec_kvcache, att1 = self.self_att.incremental(layer, step, dec_kvcache, norm1)
        # norm1 = self.norm1(next_embed, att1)
        post_add1 = next_embed + att1
        norm2 = self.norm2(post_add1)
        coeff, att2 = self.cross_att.incremental(layer, step, enc_kvcache, norm2)
        xattn = xattn + coeff

        post_add2 = post_add1 + att2
        norm3 = self.norm3(post_add2)
        ff = self.ff(norm3)
        out = post_add2 + ff
        # out = self.norm3(norm2, ff)
        return dec_kvcache, xattn, out

class Decoder(hk.Module):
    def __init__(self, dropout_rate, arch, is_train, tok_map, pos_enc_factor,
            embed_mat=None):
        super().__init__(name='dec')
        self.is_train = is_train
        self.L = arch['L']
        self.H = arch['H']
        self.K = arch['K']
        self.tok_map = tok_map
        self.xnorm = hk.LayerNorm((2,), True, True, name='lnormx')

        if embed_mat is None:
            self.embed_mat = EmbedMatrix(self.tok_map['n_vocab'], arch['M']) 
        else:
            self.embed_mat = embed_mat

        self.embed_layer = InputEmbedding(self.embed_mat, pos_enc_factor) 
        self.layers = [DecoderLayer(dropout_rate, arch, is_train, i) for i in range(arch['L'])]
        self.scale_factor = np.sqrt(arch['M']) ** -1

    def __call__(self, enc_out, enc_seqids, dec_seqs, dec_seqids, dec_tokids):
        """
        enc_out: btm (output from encoder)
        enc_seqids: bt (identifies which sentences each enc output belongs)
        dec_seqs: bq (tokens)
        dec_seqids: bq (identify which sentences each token belongs)
        dec_tokids: bq (position of a token within a sentence)
        returns: bce
        """
        dec_embed = self.embed_layer(dec_seqs, dec_tokids)
        dec_position_mask = jnp.equal(dec_seqids, -1).astype(jnp.int32)
        qt_cross_mask = jnp.not_equal(dec_seqids[:,:,None], enc_seqids[:,None,:]).astype(jnp.int32)
        qt_self_mask = jnp.not_equal(dec_seqids[:,None,:], dec_seqids[:,:,None]).astype(jnp.int32)

        qt_causal_mask = 1.0 - jnp.tri(dec_seqids.shape[1], k=0)
        qt_self_mask = jnp.maximum(qt_self_mask, qt_causal_mask)

        # TODO: find out if this is really needed 
        enc_out = self.xnorm(enc_out)

        out = dec_embed
        for mod in self.layers:
            out = mod(enc_out, out, dec_position_mask, qt_self_mask, qt_cross_mask)

        scaled_emb_mat = self.embed_mat() * self.scale_factor
        out = jnp.einsum('bcm,tm -> bct', out, scaled_emb_mat)
        # jax.debug.print('decoder_out: {}', out[0,0:5,0:20])
        return out

    def enc_kvcache(self, enc_out):
        # return kvcache: lbhstd (s=0 means key, s=1 means val)
        kvcache = []
        for mod in self.layers:
            kvcache.append(mod.enc_kvcache(enc_out))
        return jnp.stack(kvcache)

    def infer_simple(self, enc_out, max_gen_length, temperature=1.0):
        """
        Create an inference using direct sampling with temperature.
        Each batch element of enc_kvcache can represent a distinct conditioning
        sentence, or the same sentence, or a mix.  This way, you can generate
        multiple translations for each of multiple sentences in parallel.

        Inputs:
        enc_out: btm  (output embedding from encoder)  
        max_gen_length: maximum length of tokens to generate
        temperature: value in [0, 1], 
            lower values reshape the distribution to skew towards very
            high probability samples. 

        returns: 
        dec_input: b
        logits_step: bm     (prediction logits for step)
        """
        B, _, M = enc_out.shape
        L, H, K, V = self.L, self.H, self.K, self.tok_map['n_vocab']
        S = 2 # channels for key and value
        Q = max_gen_length

        enc_kvcache = self.enc_kvcache(enc_out)
        dec_kvcache = jnp.empty((L,B,H,S,Q,K), dtype=jnp.float32)
        scaled_emb_mat = self.embed_mat() * self.scale_factor

        dec_pred = jnp.empty((B, Q), dtype=jnp.int32)
        dec_pred = dec_pred.at[:,0].set(self.tok_map['bos'])
        dec_tokids = jnp.reshape(jnp.tile(jnp.arange(Q), B), (B,Q)) 
        bos_embed = self.embed_layer(dec_pred[:,0:1], dec_tokids[:,0:1])

        def step_fn(step, val):
            dec_kvcache, dec_pred, next_embed = val
            for layer, mod in enumerate(self.layers):
                dec_kvcache, next_embed = mod.incremental(layer, step, enc_kvcache,
                        dec_kvcache, next_embed) 
            logits = jnp.einsum('bcm,vm -> bcv', next_embed, scaled_emb_mat)
            sample = jax.random.categorical(hk.next_rng_key(), logits, axis=2)
            tok_ids = jax.lax.dynamic_slice_in_dim(dec_tokids, step+1, 1, 1)
            next_embed = self.embed_layer(sample, tok_ids) 
            dec_pred = jax.lax.dynamic_update_slice(dec_pred, sample, (0,step+1))
            return dec_kvcache, dec_pred, next_embed

        init = dec_kvcache, dec_pred, bos_embed
        _, dec_pred, _ = jax.lax.fori_loop(0, Q-1, step_fn, init)

        return dec_pred

    def log_prob(self, enc_out, enc_seqids, dec_seqs):
        """
        enc_out: btm (output from encoder, padded)
        enc_seqids: bt (ids of each sequence, or -1 if non-sequence)
        dec_seqs:     bq  (decoder tokens)
        Compute log P(dec|enc_out)
        """
        B,T,_ = enc_out.shape
        _,Q = dec.shape
        dec_seqids = jnp.zeros((B,Q), dtype=jnp.int32)
        dec_tokids = jnp.repeat(jnp.arange(Q)[:,None], B, axis=0)
        dec_out = self(enc_out, enc_seqids, dec_seqs, dec_seqids, dec_tokids)

        dec_probs = jax.nn.log_softmax(dec_out, axis=2) # bqv

        inds = jnp.concat((jnp.arange(B)[:,None,None], dec_tokids), axis=2) # b,q,2
        gd = jax.lax.GatherDimensionNumbers((), (0,1), (0,1,2))
        terms = jax.lax.gather(dec_probs, inds, gd, (1,1))
        return terms.sum(axis=1)

    def logits_step(self, enc_kvcache, step, dec_kvcache, xattn, new_toks):
        """
        step: token position of new_toks
        enc_kvcache: lbhstd  (encoder cache precomputed from some input)
        dec_kvcache: lbhsqd  (decoder cache populated up to q=step-1)
        xattn: bt
        new_toks: b  (batch of token ids at position step)

        returns:
        dec_kvcache, xattn updated with information at position step
        logits:  for choosing tokens at position step+1
        """
        # print(f'Decoder::logits_step: '
                # f'{enc_kvcache.shape=} {dec_kvcache.shape=} {xattn.shape=} '
                # f'{new_toks.shape=}')
        tok_ids = jnp.full_like(new_toks[:,None], step)
        new_embed = self.embed_layer(new_toks[:,None], tok_ids)

        for layer, mod in enumerate(self.layers):
            dec_kvcache, xattn, new_embed = mod.incremental(layer, step,
                    enc_kvcache, dec_kvcache, xattn, new_embed) 

        scaled_emb_mat = self.embed_mat() * self.scale_factor
        logits = jnp.einsum('bcm,vm -> bcv', new_embed, scaled_emb_mat)
        return dec_kvcache, xattn, logits 

    def beam_search(self, enc_out, alpha, beta, beam_size, max_length):
        """
        Inputs:
        enc_out: btm  (output embedding from encoder)  

        See funcs.beam_search_step for details
        """
        B, C, M = enc_out.shape
        L, H, K, V = self.L, self.H, self.K, self.tok_map['n_vocab']
        O, E = beam_size, 2 * beam_size
        S = 2 # channels for key and value
        Q = max_length

        enc_kvcache = self.enc_kvcache(enc_out)
        enc_kvcache = jnp.repeat(enc_kvcache, E, 1)
        beam_step_fn = functools.partial(funcs.beam_search_step, self.tok_map['eos'],
                alpha, beta, beam_size)
        logits_fn = functools.partial(self.logits_step, enc_kvcache)

        def bsearch_loop_fn(step, args):
            """
            all loop variables are populated in [:step]
            live_seqs, live_scores are populated in [:step+1]
            Returns: updated version of args
            """
            dec_kvcache, xattn, live_seqs, *rest = args
            new_toks = jax.lax.dynamic_slice_in_dim(live_seqs, step, 1, 2).ravel()
            # fills in next step info for dec_kvcache and xattn, and computes logits
            dec_kvcache = dec_kvcache.reshape(L,B*E,H,S,Q,K)
            xattn = xattn.reshape(B*E,C)

            # populates xattn and dec_kvcache at position step
            # produces logits for position step+1 
            dec_kvcache, xattn, logits = logits_fn(step, dec_kvcache, xattn, new_toks)

            dec_kvcache = dec_kvcache.reshape(L,B,E,H,S,Q,K)
            xattn = xattn.reshape(B,E,C)
            logits = logits.reshape(B,E,V)

            # produces  
            return beam_step_fn(step+1, logits, dec_kvcache, xattn, live_seqs, *rest)

        # these two are local to the decoding step so have a flattened B*O batch dim
        dec_kvcache = jnp.empty((L,B,E,H,S,Q,K), dtype=jnp.float32)
        xattn = jnp.zeros((B,E,C), dtype=jnp.float32)

        live_seqs = jnp.zeros((B,E,Q), dtype=jnp.int32)
        live_seqs = live_seqs.at[:,0,0].set(self.tok_map['bos'])
        live_scores = jnp.full((B,E), -jnp.inf)
        live_scores = live_scores.at[:,0].set(0.0)
        fin_seqs = jnp.zeros((B,O,Q), dtype=jnp.int32)
        fin_scores = jnp.full((B,O), -jnp.inf)
        inits = dec_kvcache, xattn, live_seqs, live_scores, fin_seqs, fin_scores 
        outs = jax.lax.fori_loop(0, max_length, bsearch_loop_fn, inits)
        _, _, live_seqs, live_scores, fin_seqs, fin_scores = outs
        # jax.debug.print('Final: fin_seqs:\n{}', fin_seqs)
        return fin_seqs, fin_scores


class Model(hk.Module):
    def __init__(self, dropout_rate, pos_enc_factor, arch, is_train, tok_map):
        super().__init__(name='tx')
        self.is_train = is_train
        self.L = arch['L']  
        self.H = arch['H']
        self.tok_map = tok_map
        self.embed_mat = EmbedMatrix(self.tok_map['n_vocab'], arch['M']) 
        # self.embed_layer = InputEmbedding(self.embed_mat, pos_enc_factor) 
        self.encoder = Encoder(dropout_rate, arch, is_train, pos_enc_factor,
                self.embed_mat)
        self.decoder = Decoder(dropout_rate, arch, is_train, self.tok_map,
                pos_enc_factor, self.embed_mat)

    def batch(self, inputs, targets):
        """
        inputs and targets are produced by pack.pack_dataset
        each is a map as follows:
            seqs: bc => packed tokenized sequences
            seqids: bc => corresponding input-target pair ids for sequences
            tokids: bc => position of token within each seqid
            counts: bt => lengths of each sequence at each 'try' (see pack)

        returns: bct
        """
        if not self.is_train:
            raise RuntimeError(f'Model.__call__ is only for training')

        enc_output = self.encoder(inputs['seqs'], inputs['seqids'], inputs['tokids'])
        dec_output = self.decoder(enc_output, inputs['seqids'], 
                targets['seqs'], targets['seqids'], targets['tokids'])
        return dec_output

    def infer_simple(self, enc_tokens, max_gen_length, temperature=1.0):
        """
        enc_tokens: bt
        max_gen_length: maximum length of tokens to generate
        temperature: value in [0, 1], 
            lower values reshape the distribution to skew towards very
            high probability samples. 
        """
        if self.is_train:
            raise RuntimeError(f'Model.infer_simple called while in training mode')

        B,V = enc_tokens.shape
        enc_tokids = jnp.reshape(jnp.tile(jnp.arange(V), B), (B,V)) 
        enc_seqids = jnp.zeros_like(enc_tokids, dtype=jnp.int32)
        enc_out = self.encoder(enc_tokens, enc_seqids, enc_tokids)
        return self.decoder.infer_simple(enc_out, max_gen_length, temperature)
        
    def embed_to_score(self, enc_embed, enc_seqids, dec_seqs):
        """
        Computes log P(dec_seqs|enc_embed)
        Used for computing output score receptivity to each input position
        during beam search

        enc_embed: btm (input embedding, as output from encoder.embed_layer 
        dec_seqs: bq (decoder tokens, -1 for missing
        """
        enc_out = self.encoder.from_embedding(enc_embed, enc_seqids)
        return self.decoder.log_prob(enc_out, enc_seqids, dec_seqs)

    """
    From the authors of https://arxiv.org/pdf/1609.08144.pdf: 

     Firstly, at each step, we only consider tokens that have local scores that are not
     more than beamsize below the best token for this step. Secondly, after a normalized
     best score has been found according to equation 14, we prune all hypotheses that are
     more than beamsize below the best normalized score so far. The latter type of
     pruning only applies to full hypotheses because it compares scores in the normalized
     space, which is only available when a hypothesis ends. This latter form of pruning
     also has the effect that very quickly no more hypotheses will be generated once a
     sufficiently good hypothesis has been found, so the search will end quickly

    """
    def beam_search(self, enc_tokens, alpha, beta, beam_size, max_gen_length):
        """
        enc_tokens: bt
        alpha, beta, beam_size, max_gen_length: parameters for beam search
        """
        B,T = enc_tokens.shape
        enc_tokids = jnp.reshape(jnp.tile(jnp.arange(T), B), (B,T)) 
        enc_seqids = jnp.zeros_like(enc_tokids, dtype=jnp.int32)
        enc_out = self.encoder(enc_tokens, enc_seqids, enc_tokids)
        jax.debug.print('Model: beam_search: enc_out.shape: {}, enc_out[0,:,0:5]\n{}', 
                enc_out.shape, enc_out[0,:,0:5])
        return self.decoder.beam_search(enc_out, alpha, beta, beam_size, max_gen_length)

    def total_params(self):
        # get total number of parameters
        return sum(par.size for par in self.params_dict().values())

    def param_shape_map(self):
        from collections import Counter
        shape_map = Counter(tuple(par.shape) for par in self.params_dict().values())
        return

    def dec_enc_attention(self, enc_kvcache, dec_input):
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
    def __init__(self, token_histo, bos_id, smoothing_eps=0.1):
        """
        page 8, Label Smoothing: using eps = 0.1
        """
        super().__init__()
        # self.eps = smoothing_eps
        self.bos_id = bos_id
        token_histo = token_histo.astype(jnp.float32)
        self.token_dist = token_histo / jnp.sum(token_histo)
        self.V = self.token_dist.shape[0]
        self.eps = smoothing_eps

    def __call__(self, dec_input, dec_output_logits):
        """
        dec_input: bq
        dec_output_logits: bqv
        """
        # bc
        targets = dec_input[:,1:]

        # -1 is non-sample, bos_id represents the start of a new sample
        # both of which should not be active as a target
        targets_active = jnp.logical_and(
                jnp.not_equal(targets, self.bos_id),
                jnp.not_equal(targets, -1))

        # bqv
        targets = jax.nn.one_hot(targets, self.V, axis=2)
        targets = (1.0 - self.eps) * targets + self.eps * self.token_dist[None,None,:]

        # jax.debug.print('{}', dec_mask)
        dec_pred_logits = dec_output_logits[:,:-1,:]

        # dec_pred = jax.nn.softmax(dec_pred_logits, axis=2)
        kldiv = funcs.fused_kldiv_softmax(targets, dec_pred_logits, 2)
        mean_kldiv = kldiv.mean(where=targets_active)

        cross_entropy = funcs.cross_entropy(targets, dec_pred_logits, 2)
        mean_cross_entropy = cross_entropy.mean(where=targets_active)

        label_entropy = funcs.entropy(targets, axis=2).mean(where=targets_active)
        return mean_kldiv, label_entropy, mean_cross_entropy


def _wrap_haiku(mod_cls, *args):
    # This is convenient if you just want to call the '__call__' method of the module
    def wrapped_fn(*call_args):
        mod = mod_cls(*args)
        return mod(*call_args)
    return wrapped_fn

def make_train_model(hps, tok_map):
    arch = dict(zip('HMKVFL', (hps.H, hps.M, hps.K, hps.V, hps.F, hps.num_layers)))
    args = hps.dropout_rate, hps.pos_encoding_factor, arch, True, tok_map 
    def wrap_fn(*call_args):
        mod = Model(*args)
        return mod.batch(*call_args)
    return hk.transform(wrap_fn)

def make_test_model(hps, tok_map):
    arch = dict(zip('HMKVFL', (hps.H, hps.M, hps.K, hps.V, hps.F, hps.num_layers)))
    args = hps.dropout_rate, hps.pos_encoding_factor, arch, False, tok_map 
    def wrap_fn(*call_args):
        mod = Model(*args)
        return mod.beam_search(*call_args)
    return hk.transform(wrap_fn)

def make_score_model(hps, tok_map):
    arch = dict(zip('HMKVFL', (hps.H, hps.M, hps.K, hps.V, hps.F, hps.num_layers)))
    args = hps.dropout_rate, hps.pos_encoding_factor, arch, False, tok_map 
    def wrap_fn(*call_args):
        mod = Model(*args)
        return mod.embed_to_score(*call_args)
    return hk.transform(wrap_fn, apply_rng=False)

def make_test_objective(hps, token_info):
    histo = token_info['histo']
    mask_id = token_info['mask']
    return hk.transform(_wrap_haiku(Objective, histo, mask_id, hps.label_smooth_eps))

def make_objective(hps, token_info):
    histo = token_info['histo']
    bos_id = token_info['bos']
    return hk.transform(_wrap_haiku(Objective, histo, bos_id, hps.label_smooth_eps))

def make_grads(cls, inst_args, out_grad, call_args):
    """
    Show the gradients passed down by cls when called with call_args.
    out_shape:  shape of this module's output when called with call_args
    """
    rng_key = jax.random.PRNGKey(42)
    layer = hk.transform(_wrap_haiku(cls, *inst_args))
    params = layer.init(rng_key, *call_args)
    primal, vjp_fn = jax.vjp(layer.apply, params, rng_key, *call_args)
    return vjp_fn(out_grad)

