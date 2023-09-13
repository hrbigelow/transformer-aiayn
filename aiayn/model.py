import functools
from dataclasses import dataclass
import re
import numpy as np
from . import hparams
from . import funcs
from . import data
import pdb
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk

class MultiHeadAttention(hk.Module):
    """
    Implements Multi-head attention from section 3.2.2
    """
    def __init__(self, H, M, K, V, do_cache=False):
        super().__init__(name='att')
        self.H = H # number of heads
        self.M = M # d_model, or 'embedding dimension'
        self.K = K # number of components in the key
        self.V = V # number of components in the value
        self.do_cache = do_cache
        self.mscale = np.sqrt(self.M) ** -1
        self.vscale = np.sqrt(self.V) ** -1
        self.kscale = np.sqrt(self.K) ** -1

    def __call__(self, qinput, kvinput, qtmask):
        """
        all masks have 0 or 1.  1 means ignore (mask out)
        qinput: bqm 
        kvinput: btm 
        qtmask: bqt  (ignore combinations of query, target positions)
        output: 
           mha: bqm
           attention matrix: bqt
        """
        B,Q,_ = qinput.shape
        _,T,_ = kvinput.shape

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
        
        alogit = jnp.einsum('bhqd,bhtd->bhqt', query, key)
        active = jnp.broadcast_to(jnp.logical_not(qtmask)[:,None,:,:], alogit.shape)
        # att2 = jax.nn.softmax(alogit * self.kscale, axis=3, where=active,
                # initial=0.0)
        att = funcs.softmax(alogit * self.kscale, axis=3, where=active, initial=0.0)
        pre = jnp.einsum('bhqt,bhtd->bhqd', att, val)
        out = jnp.einsum('hdm,bhqd->bqm', wo, pre)
        # jax.debug.print('qinput: {}\nkvinput: {}\n', qinput, kvinput)
        return out, att

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

    def incremental(self, layer, step, kvcache, new_toks, tmask):
        """
        kvcache: lbhstd (s=0 means key, s=1 means val) (from encoder or decoder)
        new_toks: b1m
        tmask: bt
        i: index into kcache and vcache where new_toks resides

        returns a tuple of:
        kvcache (updated at position i)
        out: b1m 
        """
        _,B,_,_,T,_ = kvcache.shape

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

        if self.do_cache:
            wkv = jnp.concatenate((wk[:,:,None,:], wv[:,:,None,:]), 2)
            kv_next = jnp.einsum('hmsd,bm->bhsd', wkv, new_toks)
            kv_next_unsq = kv_next[None,:,:,:,None,:]
            kvcache = jax.lax.dynamic_update_slice(kvcache, kv_next_unsq, (layer,0,0,0,step,0))

        kcache_layer = kvcache[layer,:,:,0,:,:]
        alogit = jnp.einsum('bhd,bhtd->bht', query, kcache_layer)
        logits_mask = tmask[:,None,:]
        active = jnp.broadcast_to(1 - logits_mask, alogit.shape)
        # if not self.self_attn and step == 5:
            # jax.debug.print('step:\n{}\nactive:\n{}\n', step, active)

        # if self.self_attn:
            # jax.debug.print('step {}\nself: active incremental:\n{}\n', step,
                    # active[0,0,:])
        att = funcs.softmax(alogit * self.kscale, axis=2, where=active, initial=0.0)
        # att_nomask = funcs.softmax(alogit * self.kscale, axis=2)

        # if not self.self_attn:
            # jax.debug.print('step: {}\nlayer: {}\natt[0,:,:]\n{}\natt_nomask[0,:,:]\n{}\n',
                    # step, layer, att[0,:,:], att_nomask[0,:,:])

        # if not self.self_attn and layer == 1:
            # jax.debug.print('step: {}\nquery:\n{}\natt[0,0,:]:\n{}\nactive[0,0,:]:\n{}\n', 
                    # step, query[0,0,0:10], att[0,0,:], active[0,0,:])

        # att = jax.nn.softmax(attn_logit * self.kscale, axis=2)
        pre = jnp.einsum('bht,bhtd->bhd', att, kvcache[layer,:,:,1])
        out = jnp.einsum('hdm,bhd->bm', wo, pre)

        coeff_summary = att.mean(axis=1) # mean over heads 

        if self.do_cache: 
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

    def __call__(self, input):
        """
        input: bqm
        qmask: bq
        returns: bqm
        """
        dtype = input.dtype
        w1_init = hk.initializers.RandomNormal(self.mscale, 0.0)
        w2_init = hk.initializers.RandomNormal(self.fscale, 0.0)
        w1 = hk.get_parameter('w1', [self.M,self.F], dtype, w1_init)
        b1 = hk.get_parameter('b1', [self.F], dtype, jnp.zeros)
        w2 = hk.get_parameter('w2', [self.F,self.M], dtype, w2_init)
        b2 = hk.get_parameter('b2', [self.M], dtype, jnp.zeros)
        s = jax.nn.relu(jnp.einsum('mf, bqm -> bqf', w1, input) + b1)
        out = jnp.einsum('fm, bqf -> bqm', w2, s) + b2
        # jax.debug.print('ff_out: {}', out)
        return out

class EmbedMatrix(hk.Module):
    def __init__(self, V, M):
        super().__init__(name='embed_matrix')
        self.V = V
        self.M = M

    def __call__(self):
        scale = self.M ** -0.5
        init = hk.initializers.RandomNormal(scale, 0.0)
        return hk.get_parameter('emb', [self.V, self.M], np.float32, init) 

class InputEmbedding(hk.Module):
    def __init__(self, hps, embed_mat, matrix_scale_factor, is_train):
        super().__init__(name='emb')
        self.embed_mat = embed_mat
        self.mat_factor = matrix_scale_factor
        self.pos_factor = hps.pos_encoding_factor
        self.dropout_rate = hps.dropout_rate
        self.is_train = is_train

    def positional_embedding(self, tokids):
        # tokids: bc
        denom = 10000 ** jnp.linspace(0, 1, self.embed_mat.M)
        arg = tokids[:,:,None] / denom[None,None,:]
        # it shouldn't matter what order embeddings are placed but here I follow
        # the paper, and alternate sin with cos
        pos_emb = jnp.empty_like(arg)
        pos_emb = pos_emb.at[:,::2].set(jnp.sin(arg[:,::2]))
        pos_emb = pos_emb.at[:,1::2].set(jnp.cos(arg[:,1::2]))
        return pos_emb

    def __call__(self, seqs, tokids):
        """
        pack['seqs']: bc (end-to-end packed tokens from sentences)
        pack['tokids']: bc (ids identifying position of token in a sentence, or -1 if masked)
        output: bcm
        """
        pos_embed = self.positional_embedding(tokids)
        # jax.debug.print('pos_factor: {}, pos_embed[0]:\n{}', self.pos_factor, pos_embed[0])
        # jax.debug.print('tokids: {}', tokids[0])
        embed = jnp.take(self.embed_mat(), seqs, axis=0)
        embed = jnp.where(jnp.equal(tokids, -1)[:,:,None], 0, embed)
        
        # embed = funcs.take(self.embed_mat(), input.astype(jnp.float32), axis=0)
        scaled_embed = embed * self.mat_factor
        # mean_embed_norm = jnp.sqrt((scaled_embed ** 2).sum(axis=2)).mean()
        # mean_pos_norm = jnp.sqrt((pos_embed ** 2).sum(axis=2)).mean()
        # jax.debug.print('mean_embed_norm: {}\nmean_pos_norm: {}\n',
                # mean_embed_norm, mean_pos_norm)
        # full_embed = scaled_embed + pos_embed
        full_embed = scaled_embed + pos_embed * self.pos_factor
        if self.is_train:
            full_embed = hk.dropout(hk.next_rng_key(), self.dropout_rate, full_embed)
        return full_embed

class EncoderLayer(hk.Module):
    def __init__(self, hps, arch, is_train, layer_num):
        super().__init__(name=f'layer{layer_num:02d}')
        H, M, K, V, F = tuple(arch[l] for l in 'HMKVF') 
        self.layer_num = layer_num
        self.is_train = is_train
        self.dropout_rate = hps.dropout_rate
        self.attention = MultiHeadAttention(H, M, K, V, do_cache=True)
        self.norm1 = hk.LayerNorm((2,), True, True, name='lnorm1')
        self.norm2 = hk.LayerNorm((2,), True, True, name='lnorm2')
        self.ff = PositionwiseFF(M, F)

    def __call__(self, input_embed, qtmask):
        """
        input_embed: btm
        pack: pack from aiayn.pack
        qtmask: bqt (bool)
        returns: 
        output_embedding: bqm
        attn_entropy: b

        Architecture is the pre-LN Layer as described in https://arxiv.org/pdf/2002.04745.pdf
        but with additional dropout added in the same position as in https://arxiv.org/abs/1706.03762

        In my experiments, post-LN resulted in homogenization of the encoder
        embeddings (all embedding vectors at every position were nearly identical)
        """
        norm1 = self.norm1(input_embed)
        att, att_coeff = self.attention(norm1, norm1, qtmask)
        if self.is_train:
            att = hk.dropout(hk.next_rng_key(), self.dropout_rate, att)
        post_add1 = input_embed + att
        norm2 = self.norm2(post_add1)

        ff = self.ff(norm2)
        if self.is_train:
            ff = hk.dropout(hk.next_rng_key(), self.dropout_rate, ff)
        out = post_add1 + ff 
        return out, att_coeff

class Encoder(hk.Module):
    def __init__(self, hps, arch, is_train, embed_mat):
        super().__init__(name='enc')
        self.is_train = is_train
        self.embed_layer = InputEmbedding(hps, embed_mat, jnp.sqrt(arch['M']), is_train) 
        self.layers = [EncoderLayer(hps, arch, is_train, i) for i in range(arch['L'])]

    def __call__(self, inputs, targets=None):
        """
        pack:  sentence-pair format from aiayn/pack.py
        returns
        output embedding: bqm
        attention entropy: bl
        """
        tmask = jnp.equal(inputs['tokids'], -1)
        qtmask = jnp.not_equal(inputs['seqids'][:,None,:], inputs['seqids'][:,:,None])
        qtmask = jnp.logical_or(qtmask, tmask[:,None,:])
        qtmask = jnp.logical_or(qtmask, tmask[:,:,None])

        input_embed = self.embed_layer(inputs['seqs'], inputs['tokids']) 
        # jax.debug.print('input_embed[:,:,0:2]:\n{}', input_embed[:,:,0:2])
        out = input_embed
        # pdb.set_trace()
        attn_entropies = []
        for mod in self.layers:
            out, attn_coeff = mod(out, qtmask)
            if self.is_train and self.with_attn_entropy:
                att_query, att_target, att_output = inputs, inputs, targets 
                attn_entropy = funcs.attention_entropy(
                        att_coeff, qtmask,
                        att_query['seqids'], att_query['counts'],
                        att_target['seqids'], att_target['counts'],
                        att_output['counts'])
                attn_entropies.append(attn_entropy)
            else:
                attn_entropies.append(jnp.zeros(out.shape[0]))

        # jax.debug.print('encoder_out: {}', out)
        # jax.debug.print('attn_entropy {}', attn_entropies[0])
        attn_entropy_all = jnp.stack(attn_entropies, axis=1) # bl
        # jax.debug.print('attn_entropy_all:\n{}\n', attn_entropy_all)
        return out, attn_entropy_all

    def from_embedding(self, embed, seqids):
        """
        embed: btm  (input embedding)
        seqids: bt  (ids grouping tokens into separate sequences, -1 for non-sample)
        """
        # TODO: fix this
        position_mask = jnp.equal(seqids, -1)
        tmask = jnp.equal(pack['tokids'], -1)
        qtmask = jnp.not_equal(seqids[:,None,:], seqids[:,:,None]).astype(jnp.int32)

        out = embed
        for mod in self.layers:
            out = mod(out, position_mask, qtmask)
        return out

class DecoderLayer(hk.Module):
    def __init__(self, hps, arch, is_train, layer_num):
        super().__init__(name=f'layer{layer_num:02d}')
        H, M, K, V, F = tuple(arch[l] for l in 'HMKVF') 
        self.dropout_rate = hps.dropout_rate
        self.is_train = is_train
        self.self_attention = MultiHeadAttention(H, M, K, V, do_cache=True)
        self.cross_attention = MultiHeadAttention(H, M, K, V)
        self.ff = PositionwiseFF(M, F)
        self.norm1 = hk.LayerNorm((2,), True, True, name='lnorm1')
        self.norm2 = hk.LayerNorm((2,), True, True, name='lnorm2')
        self.norm3 = hk.LayerNorm((2,), True, True, name='lnorm3')

    def __call__(self, enc_out, input_emb, self_qtmask, cross_qtmask):
        """
        all masks have 0 or 1 (1 = mask out)
        enc_out: btm (t from encoder)
        input_emb: bqm 
        self_qtmask: bqt
        cross_qtmask: bqt
        out: bqm
        """
        norm1 = self.norm1(input_emb) 
        self_att, _ = self.self_attention(norm1, norm1, self_qtmask)
        if self.is_train:
            self_att = hk.dropout(hk.next_rng_key(), self.dropout_rate, self_att)
        post_add1 = input_emb + self_att
        norm2 = self.norm2(post_add1)

        cross_att, cross_att_coeff = self.cross_attention(norm2, enc_out, cross_qtmask)
        if self.is_train:
            cross_att = hk.dropout(hk.next_rng_key(), self.dropout_rate, cross_att)
        post_add2 = post_add1 + cross_att
        norm3 = self.norm3(post_add2)
        ff = self.ff(norm3)
        if self.is_train:
            ff = hk.dropout(hk.next_rng_key(), self.dropout_rate, ff)
        out = post_add2 + ff
        return out, cross_att_coeff


    def enc_kvcache(self, enc_out):
        """
        Compute the keys and values of the encoder output for this decoder layer
        return bhstd  (s in [0, 1], key or val)
        """
        return self.cross_attention.get_keys_values(enc_out)

    def incremental(self, layer, step, enc_mask, enc_kvcache, dec_kvcache, next_embed):
        """
        enc_mask: bt
        enc_kvcache: lbhstd 
        dec_kvcache: lbhsqd
        xattn:  bt  (cumulative attention on encoder embeddings)
        next_embed: b1m
        """
        B = dec_kvcache.shape[1]
        Q = dec_kvcache.shape[4]
        norm1 = self.norm1(next_embed)
        assert next_embed.shape[0] == enc_kvcache.shape[1]
        # print(f'{next_embed.shape=}')

        step_mask = jnp.greater(jnp.arange(Q), step).astype(jnp.int32)
        step_mask = jnp.repeat(step_mask[None,:], B, axis=0)
        # jax.debug.print('layer: {}\nstep: {}\n_mask:\n{}\n', layer, step, step_mask)
        dec_kvcache, att1 = self.self_attention.incremental(layer, step, dec_kvcache,
            norm1, step_mask)
        # norm1 = self.norm1(next_embed, att1)
        post_add1 = next_embed + att1
        norm2 = self.norm2(post_add1)
        # jax.debug.print('layer: {}\nstep: {}\nenc_mask:\n{}\n', layer, step, enc_mask)

        # enc_mask = jnp.zeros(enc_mask.shape)
        # enc_mask = enc_mask.at[:,40:].set(1)
        coeff, att2 = self.cross_attention.incremental(layer, step, enc_kvcache, norm2, enc_mask)

        # if step == 5:
            # jax.debug.print('layer: {}\nstep: {}\natt2:\n{}\n', layer, step, att2)

        post_add2 = post_add1 + att2
        norm3 = self.norm3(post_add2)
        ff = self.ff(norm3)
        out = post_add2 + ff
        # out = self.norm3(norm2, ff)
        return dec_kvcache, coeff, out

class Decoder(hk.Module):
    def __init__(self, hps, arch, is_train, bos_id, eos_id, n_vocab, embed_mat=None):
        super().__init__(name='dec')
        self.is_train = is_train
        self.L = arch['L']
        self.H = arch['H']
        self.K = arch['K']
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.n_vocab = n_vocab
        self.xnorm = hk.LayerNorm((2,), True, True, name='lnormx')

        if embed_mat is None:
            self.embed_mat = EmbedMatrix(self.n_vocab, arch['M']) 
        else:
            self.embed_mat = embed_mat

        self.embed_layer = InputEmbedding(hps, self.embed_mat, jnp.sqrt(arch['M']), is_train) 
        self.layers = [DecoderLayer(hps, arch, is_train, i) for i in range(arch['L'])]
        self.mscale = np.sqrt(arch['M']) ** -1
        self.norm = hk.LayerNorm((2,), True, True, name='lnorm')

    def __call__(self, enc_out, pack):
        """
        enc_out          : btm (output from encoder)
        pack: sentence pair pack from aiayn/pack.py
        returns: bce
        """
        inputs, targets = pack['inputs'], pack['targets']
        dec_embed = self.embed_layer(targets['seqs'], targets['tokids'])

        dec_position_mask = jnp.equal(targets['tokids'], -1)
        cross_qtmask = jnp.not_equal(targets['seqids'][:,:,None], inputs['seqids'][:,None,:])
        cross_qtmask = jnp.logical_or(cross_qtmask, dec_position_mask[:,:,None])

        causal_mask = jnp.logical_not(jnp.tri(targets['seqids'].shape[1], k=0, dtype=bool))
        self_qtmask = jnp.not_equal(targets['seqids'][:,:,None], targets['seqids'][:,None,:])
        self_qtmask = jnp.logical_or(self_qtmask, causal_mask[None,:,:])
        self_qtmask = jnp.logical_or(self_qtmask, dec_position_mask[:,:,None])
        self_qtmask = jnp.logical_or(self_qtmask, dec_position_mask[:,None,:])

        # TODO: find out if this is really needed 
        enc_out = self.xnorm(enc_out)

        attn_entropies = []
        out = dec_embed
        for mod in self.layers:
            out, att_coeff = mod(enc_out, out, self_qtmask, cross_qtmask)
            if self.is_train and self.with_attn_entropy:
                att_query, att_target, att_output = pack['targets'], pack['inputs'], pack['targets']
                attn_entropy = funcs.attention_entropy(
                        att_coeff, cross_qtmask,
                        att_query['seqids'], att_query['counts'],
                        att_target['seqids'], att_target['counts'],
                        att_output['counts'])
                attn_entropies.append(attn_entropy)
            else:
                attn_entropies.append(jnp.zeros(out.shape[0]))
        attn_entropy_all = jnp.stack(attn_entropies, axis=1) # bl

        # As shown in: 
        # https://arxiv.org/pdf/1904.10509.pdf
        # There is one final layer norm
        out = self.norm(out)

        out = jnp.einsum('bcm,tm -> bct', out, self.embed_mat())
        # jax.debug.print('decoder_out: {}', out[0,0:5,0:20])
        return out, attn_entropy_all

    def enc_kvcache(self, enc_out):
        """
        Computes the kvcache for the cross-attention modules 
        return kvcache: lbhstd (s=0 means key, s=1 means val)
        """
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
        L, H, K, V = self.L, self.H, self.K, self.n_vocab
        S = 2 # channels for key and value
        Q = max_gen_length

        enc_kvcache = self.enc_kvcache(enc_out)
        dec_kvcache = jnp.empty((L,B,H,S,Q,K), dtype=jnp.float32)

        dec_pred = jnp.empty((B, Q), dtype=jnp.int32)
        dec_pred = dec_pred.at[:,0].set(self.bos_id)
        dec_tokids = jnp.reshape(jnp.tile(jnp.arange(Q), B), (B,Q)) 
        bos_embed = self.embed_layer(dec_pred[:,0:1], dec_tokids[:,0:1])

        def step_fn(step, val):
            dec_kvcache, dec_pred, next_embed = val
            for layer, mod in enumerate(self.layers):
                dec_kvcache, next_embed = mod.incremental(layer, step, enc_mask, enc_kvcache,
                        dec_kvcache, next_embed) 
            logits = jnp.einsum('bcm,vm -> bcv', next_embed, self.embed_mat())
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
        _,Q = dec_seqs.shape
        dec_seqids = jnp.zeros((B,Q), dtype=jnp.int32)
        dec_tokids = jnp.repeat(jnp.arange(Q)[:,None], B, axis=0)
        dec_out, _ = self(enc_out, enc_seqids, dec_seqs, dec_seqids, dec_tokids)

        # dec_probs = jax.nn.log_softmax(dec_out, axis=2) # bqv
        dec_probs = funcs.log_softmax(dec_out, axis=2) # bqv

        inds = jnp.concat((jnp.arange(B)[:,None,None], dec_tokids), axis=2) # b,q,2
        gd = jax.lax.GatherDimensionNumbers((), (0,1), (0,1,2))
        terms = jax.lax.gather(dec_probs, inds, gd, (1,1))
        return terms.sum(axis=1)

    def logits_step(self, step, enc_mask, enc_kvcache, dec_kvcache, new_toks):
        """
        step: token position of new_toks
        enc_mask: bet
        enc_kvcache: lbehstd  (encoder cache precomputed from some input)
        dec_kvcache: lbehsqd  (decoder cache populated up to q=step-1)
        new_toks: b  (batch of token ids at position step)

        returns:
        dec_kvcache, xattn updated with information at position step
        logits:  bev  for choosing tokens at position step+1
        xattn: bet  cumulative normalized attention
        """
        # print(f'Decoder::logits_step: '
                # f'{enc_kvcache.shape=} {dec_kvcache.shape=} {xattn.shape=} '
                # f'{new_toks.shape=}')
        tok_ids = jnp.full_like(new_toks[:,None], step)
        new_embed = self.embed_layer(new_toks[:,None], tok_ids) # b1m

        dsh = dec_kvcache.shape
        esh = enc_kvcache.shape
        msh = enc_mask.shape

        enc_mask_flat = jnp.reshape(enc_mask, (msh[0] * msh[1], msh[2]))
        dec_kvcache_flat = jnp.reshape(dec_kvcache, (dsh[0], dsh[1] * dsh[2], *dsh[3:]))
        enc_kvcache_flat = jnp.reshape(enc_kvcache, (esh[0], esh[1] * esh[2], *esh[3:]))

        xattn = jnp.zeros_like(enc_mask_flat)
        for layer, mod in enumerate(self.layers):
            dec_kvcache_flat, xattn_layer, new_embed = mod.incremental(layer, step,
                    enc_mask_flat, enc_kvcache_flat, dec_kvcache_flat, new_embed) 
            xattn = xattn + xattn_layer
        dec_kvcache = jnp.reshape(dec_kvcache_flat, dsh)
        xattn = xattn / self.L

        xattn = jnp.reshape(xattn, msh)
        #enc_kvcache = jnp.reshape(enc_kvcache_flat, esh)

        # See https://arxiv.org/pdf/1904.10509.pdf
        new_embed = self.norm(new_embed)

        logits = jnp.einsum('bm,vm -> bv', new_embed[:,0,:], self.embed_mat())
        # print(f'Before: {logits.shape=}')
        logits = jnp.reshape(logits, (*msh[:2], logits.shape[1]))
        # print(f'After: {logits.shape=}')
        return dec_kvcache, xattn, logits 

    def bsearch_loop_fn(self, beam_search_pars, enc_mask, enc_kvcache, step, dec_kvcache,
            xattn_cumul, live_seqs, *rest):
        """
        enc_mask: bet
        all loop variables are populated in [:step]
        live_seqs, live_scores are populated in [:step+1]
        xattn_cumul:  bet  (cumulative xattn from step 0 until now)
        Returns: updated version of args
        """
        new_toks = jax.lax.dynamic_slice_in_dim(live_seqs, step, 1, axis=2).ravel()
        # fills in next step info for dec_kvcache and xattn, and computes logits
        # populates xattn and dec_kvcache at position step
        # produces logits for position step+1 
        dec_kvcache, xattn_step, logits = self.logits_step(step, enc_mask, enc_kvcache,
                dec_kvcache, new_toks)
        # jax.debug.print('step {}\nlogits\n{}\n', step, logits[0,0,:10])
        # print('shapes: ', enc_mask.shape, xattn.shape)
        xattn_cumul = xattn_cumul + xattn_step
        # jax.debug.print('step {}\nxattn_cumul:\n{}\n', step, xattn_cumul)

        beam_step_data = step+1, enc_mask, logits, dec_kvcache, xattn_cumul, live_seqs, *rest
        return funcs.beam_search_step(*beam_search_pars, *beam_step_data)

    def beam_search(self, enc_mask, enc_out, alpha, beta, beam_size, max_source_len,
            max_target_len):
        """
        Inputs:
        enc_mask: bt
        enc_out: btm  (output embedding from encoder)  

        See funcs.beam_search_step for details
        """
        B, C, M = enc_out.shape
        L, H, K, V = self.L, self.H, self.K, self.n_vocab
        O, E = beam_size, 2 * beam_size
        S = 2 # channels for key and value
        Q = max_target_len # target sentence (in the decoder)
        T = max_source_len # source sentence (in the encoder)
        dtype = enc_out.dtype

        enc_out = self.xnorm(enc_out)
        enc_kvcache = self.enc_kvcache(enc_out)
        # dec_kvcache = hk.get_state('dec_kvcache', [L,B,E,H,S,Q,K], dtype, jnp.zeros)
        enc_kvcache = jnp.repeat(enc_kvcache[:,:,None], E, 2)
        # jax.debug.print('enc_kvcache:\n{}\n', enc_kvcache[0,0,0,0,:,:10,:10])
        enc_mask = jnp.repeat(enc_mask[:,None,:], E, 1)
        # jax.debug.print('beam_search: enc_out:\n{}\n', enc_out)
        # jax.debug.print('beam_search: enc_mask:\n{}\n', enc_mask)

        # these two are local to the decoding step so have a flattened B*O batch dim
        dec_kvcache = jnp.empty((L,B,E,H,S,Q,K), dtype=jnp.float32)
        xattn = jnp.zeros((B,E,T), dtype=jnp.float32)

        live_seqs = jnp.zeros((B,E,Q), dtype=jnp.int32)
        live_seqs = live_seqs.at[:,0,0].set(self.bos_id)
        live_scores = jnp.full((B,E), -jnp.inf)
        live_scores = live_scores.at[:,0].set(0.0)
        fin_seqs = jnp.zeros((B,O,Q), dtype=jnp.int32)
        fin_scores = jnp.full((B,O), -jnp.inf)
        inits = (dec_kvcache, xattn, live_seqs, live_scores, fin_seqs, fin_scores)

        bsearch_pars = self.eos_id, alpha, beta, beam_size
        loop_fn = lambda step, vals: self.bsearch_loop_fn(bsearch_pars, enc_mask,
                enc_kvcache, step, *vals)
        outs = jax.lax.fori_loop(0, max_target_len, loop_fn, inits)
        _, _, live_seqs, live_scores, fin_seqs, fin_scores = outs
        # jax.debug.print('Final: fin_seqs:\n{}', fin_seqs)
        return fin_seqs, fin_scores 


class Model(hk.Module):
    def __init__(self, hps, arch, is_train, bos_id, eos_id, n_vocab):
        super().__init__(name='tx')
        self.is_train = is_train
        self.L = arch['L']  
        self.H = arch['H']
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.n_vocab = n_vocab # includes bos_id and eos_id but not pad_id
        self.embed_mat = EmbedMatrix(self.n_vocab, arch['M']) 
        self.encoder = Encoder(hps, arch, is_train, self.embed_mat)
        self.decoder = Decoder(hps, arch, is_train, bos_id, eos_id, n_vocab, self.embed_mat)

    def batch(self, pack):
        """
        inputs and targets are produced by pack.pack_dataset
        each is a map as follows:
            seqs: bc => packed tokenized sequences
            seqids: bc => corresponding input-target pair ids for sequences
            tokids: bc => position of token within each seqid
            counts: bt => lengths of each sequence at each 'try' (see pack)

        returns: 
        decoder output: bct
        encoder attention entropy: l
        decoder cross-attention entropy: l
        """
        enc_output, enc_attn_entropy = self.encoder(pack['inputs'], pack['targets'])
        dec_output, dec_attn_entropy = self.decoder(enc_output, pack)
        return dec_output, enc_attn_entropy.sum(axis=0), dec_attn_entropy.sum(axis=0)

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
    def beam_search(self, enc_tokens, pad_value, alpha, beta, beam_size,
            max_source_len, max_target_len):
        """
        enc_tokens: bt (tokens, padded with pad_value
        alpha, beta, beam_size, max_gen_length: parameters for beam search
        """
        B,T = enc_tokens.shape
        enc_tokids = jnp.repeat(jnp.arange(T)[None,:], B, axis=0)
        enc_tokids = jnp.where(enc_tokens == pad_value, -1, enc_tokids)
        enc_mask = jnp.where(enc_tokens == pad_value, 1, 0)
        # jax.debug.print('enc_tokids:\n{}', enc_tokids)
        enc_seqids = jnp.zeros_like(enc_tokids, dtype=jnp.int32)
        inputs = dict(seqs=enc_tokens, seqids=enc_seqids, tokids=enc_tokids)
        enc_out, _ = self.encoder(inputs)
        # jax.debug.print('Model: beam_search: enc_out[0,0:10,0:4]\n{}', enc_out[0,0:10,0:4])
        return self.decoder.beam_search(enc_mask, enc_out, alpha, beta, beam_size,
                max_source_len, max_target_len)

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

class Objective:
    def __init__(self, hps, bos_id, n_vocab):
        # self.eps = smoothing_eps
        self.bos_id = bos_id
        self.V = n_vocab 
        self.eps = hps.label_smooth_eps
        self.with_attn_entropy = hps.with_attn_entropy 
        self.attn_loss_weight = hps.attn_loss_weight

    def metrics(self, targets, dec_output_logits):
        """
        targets: item['targets'] from packed dataset
        dec_input: bq
        dec_output_logits: bqv
        """
        # bc
        targets = jax.tree_map(lambda x: x[:,1:], targets)
        targets_active = jnp.not_equal(targets['tokids'], -1)
        target_seqs = jax.nn.one_hot(targets['seqs'], self.V, axis=2) # bqv

        # From https://arxiv.org/pdf/1512.00567.pdf "we used the uniform distribution u(k)"
        target_seqs = (1.0 - self.eps) * target_seqs + self.eps * self.V ** -1 

        # jax.debug.print('{}', dec_mask)
        dec_pred_logits = dec_output_logits[:,:-1,:]

        total_active_targets = targets_active.sum()

        kldiv = funcs.fused_kldiv_softmax(target_seqs, dec_pred_logits, 2)

        sum_kldiv = kldiv.sum(where=targets_active)

        where = jnp.broadcast_to(targets_active[:,:,None], target_seqs.shape)
        cross_ent = funcs.cross_entropy(target_seqs, dec_pred_logits, 2, where)
        sum_cross_ent = cross_ent.sum(where=targets_active)
        label_ent = funcs.entropy(target_seqs, 2, where)
        sum_label_ent = label_ent.sum(where=targets_active)

        return dict(
                sum_active=total_active_targets,
                sum_kldiv=sum_kldiv,
                sum_label_entropy=sum_label_ent,
                sum_cross_entropy=sum_cross_ent
                )

    def reduce_metrics(self, metrics):
        """
        Normalize batch-summed metrics.  These are metrics that are a sum of terms,
        one per 2D batch item.  2D is (sentence, token-in-sentence)
        """
        norms = {}
        for key, val in metrics.items():
            if key == 'sum_active':
                continue
            norm_key = key[4:]
            norms[key[4:]] = val / metrics['sum_active']
        
        norms['sum_active'] = metrics['sum_active']
        # jax.debug.print('active metrics: {}', metrics['sum_active'])
        return norms

    @staticmethod
    def renorm_loss(summed_measure, sum_active):
        """
        Computes a renormalized loss from a summed measure.
        """
        per_term = summed_measure / sum_active
        per_term_loss = jnp.sum((1.0 - per_term) ** 2)
        return per_term_loss * sum_active

    def summed_loss(self, metrics):
        """
        Called inside the pmapped function to compute gradients.
        This is the sum over individual batch term losses.  Gradients produced from
        this must be normalized by total batch size.

        Awkwardly, since attention_entropy_loss is an aggregate quantity over
        batch_dim1 (token context dimension)
        """
        sum_xent = metrics['sum_cross_entropy']

        if self.with_attn_entropy:
            sum_attn_loss = metrics['sum_enc_attn_loss'] + metrics['sum_dec_attn_loss']
            return sum_xent + sum_attn_loss * self.attn_loss_weight 
        else:
            return sum_xent


def _wrap_haiku(mod_cls, *args):
    # This is convenient if you just want to call the '__call__' method of the module
    def wrapped_fn(*call_args):
        mod = mod_cls(*args)
        return mod(*call_args)
    return wrapped_fn

def make_model(hps, bos_id, eos_id, n_vocab, do_batch, do_train):
    arch = dict(zip('HMKVFL', (hps.H, hps.M, hps.K, hps.V, hps.F, hps.num_layers)))
    args = (hps, arch, do_train, bos_id, eos_id, n_vocab)
    if do_batch:
        def wrap_fn(*call_args):
            mod = Model(*args)
            return mod.batch(*call_args)
        return hk.transform(wrap_fn)
    else:
        def wrap_fn(*call_args):
            mod = Model(*args)
            return mod.beam_search(*call_args)
        return hk.transform(wrap_fn)

def make_score_model(hps, tok_map):
    arch = dict(zip('HMKVFL', (hps.H, hps.M, hps.K, hps.V, hps.F, hps.num_layers)))
    args = hps, arch, False, tok_map 
    def wrap_fn(*call_args):
        mod = Model(*args)
        return mod.embed_to_score(*call_args)
    return hk.transform(wrap_fn, apply_rng=False)

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

