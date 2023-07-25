from dataclasses import dataclass
import re
import numpy as np
from . import hparams
from . import funcs
import pdb
import jax
import jax.numpy as jnp
import haiku as hk

class MultiHeadAttention(hk.Module):
    """
    Implements Multi-head attention from section 3.2.2
    """
    def __init__(self, H, M, K, V):
        super().__init__(name='att')
        self.H = H
        self.M = M
        self.K = K
        self.V = V
        self.mscale = np.sqrt(self.M) ** -1
        self.vscale = np.sqrt(self.V) ** -1
        self.scale_factor = np.sqrt(self.K)

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

        qmask = jnp.expand_dims(qmask, 2)   # b,q,1
        tmask = jnp.expand_dims(tmask, 1)   # b,1,t
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

        # print(wq.shape, qinput.shape)
        query = jnp.einsum('hmd,bqm->bhqd', wq, qinput)
        key = jnp.einsum('hmd,btm->bhtd', wk, kvinput)
        val = jnp.einsum('hmd,btm->bhtd', wv, kvinput)
        
        # add head dimension
        # active = jnp.equal(logits_mask, 0.0)
        # active = jnp.expand_dims(active, 1)
        # alogit = jnp.einsum('bhqd,bhtd->bhqt', query, key)
        # att = jax.nn.softmax(alogit * self.scale_factor ** -1, axis=3, 
         #        where=active, initial=1.0)

        logit_adj = jnp.expand_dims(logits_mask, 1) * -1e6
        alogit = jnp.einsum('bhqd,bhtd->bhqt', query, key) + logit_adj
        att = jax.nn.softmax(alogit * self.scale_factor ** -1, axis=3)
        pre = jnp.einsum('bhqt,bhtd->bhqd', att, val)
        out = jnp.einsum('hdm,bhqd->bqm', wo, pre)
        out = out * (1.0 - qmask) # to protect gradients of masked positions
        # jax.debug.print('qinput: {}\nkvinput: {}\n', qinput, kvinput)
        return out

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

    def __call__(self, input, qmask):
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
    def __init__(self, dropout_rate, is_train):
        super().__init__(name='res')
        self.is_train = is_train
        self.rate = dropout_rate

        # TODO: What is the meaning of the 'axis' argument here?
        # If I choose 1, then the grad_tests for decoder_layer fails, causing
        # leakage across autoregressive positions
        # Tim says: only apply across embedding axis, not batch axis
        self.layer_norm = hk.LayerNorm(axis=(2,), create_scale=True, create_offset=True,
                name='lnorm')

    def __call__(self, residual, proximal, qmask):
        """
        residual: bdm
        proximal: bdm
        qmask: bd
        output: bdm
        """
        if qmask is None:
            B,D,_ = residual.shape
            qmask = jnp.zeros((B,D))

        if self.is_train:
            proximal = hk.dropout(hk.next_rng_key(), self.rate, proximal)
        add = (residual + proximal) * (1.0 - jnp.expand_dims(qmask, 2))
        # return add
        return self.layer_norm(add)

class EmbedMatrix(hk.Module):
    def __init__(self, T, M):
        super().__init__(name='embed_matrix')
        self.T = T
        self.M = M

    def __call__(self):
        init = hk.initializers.RandomNormal(1.0, 0.0)
        return hk.get_parameter('emb', [self.T, self.M], np.float32, init) 

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
        embed = jnp.take(self.embed_mat(), seq_tokens, axis=0)
        
        # embed = funcs.take(self.embed_mat(), input.astype(jnp.float32), axis=0)
        # jax.debug.print('embed: {}\npos_embed: {}\n', embed, pos_embed)
        return embed + pos_embed * self.pos_factor

class EncoderLayer(hk.Module):
    def __init__(self, dropout_rate, arch, is_train, layer_num):
        super().__init__(name=f'layer{layer_num:02d}')
        H, M, K, V, F = tuple(arch[l] for l in 'HMKVF') 
        self.att = MultiHeadAttention(H, M, K, V)
        self.norm1 = DropoutAddAndNorm(dropout_rate, is_train)
        self.ff = PositionwiseFF(M, F)
        self.norm2 = DropoutAddAndNorm(dropout_rate, is_train)

    def __call__(self, input, position_mask, qt_mask):
        """
        input: btm
        position_mask: bt
        qt_mask: bqt
        returns: bqm
        """
        att = self.att(input, input, position_mask, position_mask, qt_mask)
        # return self.ff(att, pad_mask)
        norm = self.norm1(input, att, position_mask)
        ff = self.ff(norm, position_mask)
        # jax.debug.print('encoder_layer ff: {}', ff)
        out = self.norm2(norm, ff, position_mask)
        return out

class Encoder(hk.Module):
    def __init__(self, dropout_rate, arch, is_train):
        super().__init__(name='enc')
        self.layers = [EncoderLayer(dropout_rate, arch, is_train, i) for i in range(arch['L'])]

    def __call__(self, input, sample_mask):
        """
        input: btm
        sample_mask: bt  (-1 is non-sample, >=0 are sample numbers)
        returns: bqm
        """
        position_mask = jnp.equal(sample_mask, -1).astype(jnp.int32)
        qt_mask = jnp.not_equal(
                jnp.expand_dims(sample_mask, 1),
                jnp.expand_dims(sample_mask, 2)).astype(jnp.int32)
        # print(f'Encoder: {position_mask.shape=}, {qt_mask.shape=}')
        out = input
        # jax.debug.print('out: {}', out)
        for mod in self.layers:
            out = mod(out, position_mask, qt_mask)
        # jax.debug.print('encoder_out: {}', out)
        return out

class DecoderLayer(hk.Module):
    def __init__(self, dropout_rate, arch, is_train, layer_num):
        super().__init__(name=f'layer{layer_num:02d}')
        H, M, K, V, F = tuple(arch[l] for l in 'HMKVF') 
        self.self_att = MultiHeadAttention(H, M, K, V)
        self.norm1 = DropoutAddAndNorm(dropout_rate, is_train)
        self.cross_att = MultiHeadAttention(H, M, K, V)
        self.norm2 = DropoutAddAndNorm(dropout_rate, is_train)
        self.ff = PositionwiseFF(M, F)
        self.norm3 = DropoutAddAndNorm(dropout_rate, is_train)

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
        qt_causal_mask = 1.0 - jnp.tri(Cdec, k=0)
        qt_self_mask = jnp.maximum(qt_causal_mask, qt_self_mask)

        att1 = self.self_att(input, input, position_mask, position_mask, qt_self_mask)
        norm1 = self.norm1(input, att1, position_mask)
        att2 = self.cross_att(norm1, enc_out, position_mask, None, qt_cross_mask)
        norm2 = self.norm2(norm1, att2, position_mask)
        ff = self.ff(norm2, position_mask)
        out = self.norm3(norm2, ff, position_mask)
        return out

class Decoder(hk.Module):
    def __init__(self, dropout_rate, arch, T, is_train, embed_mat=None):
        super().__init__(name='dec')
        self.is_train = is_train
        if embed_mat is None:
            self.embed_mat = EmbedMatrix(T, arch['M']) 
        else:
            self.embed_mat = embed_mat
        self.layers = [DecoderLayer(dropout_rate, arch, is_train, i) for i in range(arch['L'])]
        # self.linear_final = hk.Linear(T)
        self.scale_factor = np.sqrt(arch['M']) ** -1

    def __call__(self, enc_out, dec_in, enc_mask, dec_mask):
        """
        enc_out: btm 
        dec_in: bqm
        enc_mask: bt (-1 = non-sample.  >=0 is sample number)
        dec_mask: bq (-1 = non-sample.  >=0 is sample number)
        returns: bce
        """
        if dec_mask is not None:
            assert dec_in.shape[1] == dec_mask.shape[1], \
                    f'Decoder shape error: {dec_in.shape=}, {dec_mask.shape=}'
        if enc_mask is not None:
            assert enc_out.shape[1] == enc_mask.shape[1], \
                    f'Decoder shape error: {enc_out.shape=}, {enc_mask.shape=}'

        out = dec_in
        dec_position_mask = jnp.equal(dec_mask, -1).astype(jnp.int32)
        qt_cross_mask = jnp.not_equal(
                jnp.expand_dims(dec_mask, 2),
                jnp.expand_dims(enc_mask, 1)).astype(jnp.int32)

        qt_self_mask = jnp.not_equal(
                jnp.expand_dims(dec_mask, 1),
                jnp.expand_dims(dec_mask, 2)).astype(jnp.int32)


        for mod in self.layers:
            out = mod(enc_out, out, dec_position_mask, qt_self_mask, qt_cross_mask)
        # print(f'Decoder.__call__: {enc_out.shape=}, {out.shape=}')
        scaled_emb_mat = self.embed_mat() * self.scale_factor
        # out = self.linear_final(out)
        out = jnp.einsum('bcm,tm -> bct', out, scaled_emb_mat)
        # jax.debug.print('decoder_out: {}', out)
        # out = jax.nn.softmax(out, axis=2)
        return out

class Model(hk.Module):
    def __init__(self, dropout_rate, pos_enc_factor, arch, is_train, n_vocab, mask_id):
        super().__init__(name='tx')
        self.is_train = is_train
        self.T = n_vocab 
        self.mask_id = mask_id
        self.embed_mat = EmbedMatrix(self.T, arch['M']) 
        self.embed_layer = InputEmbedding(self.embed_mat, pos_enc_factor) 
        self.encoder = Encoder(dropout_rate, arch, is_train)
        self.decoder = Decoder(dropout_rate, arch, is_train, self.T, self.embed_mat)

    def __call__(self, inputs, targets, temperature=1.0):
        """
        inputs and targets are produced by pack.pack_dataset
        seqs: bc => packed tokenized sequences
        seqids: bc => corresponding input-target pair ids for sequences
        tokids: bc => position of token within each seqid
        counts: bt => lengths of each sequence at each 'try' (see pack)

        enc_input: bc
        dec_input: bc
        enc_mask: bc
        dec_mask: bc
        returns: bct
        """
        enc_embed = self.embed_layer(inputs['seqs'], inputs['tokids'])
        dec_embed = self.embed_layer(targets['seqs'], targets['tokids'])
        # print(f'{enc_input.shape=}, {enc_mask.shape=}')

        if self.is_train:
            enc_output = self.encoder(enc_embed, inputs['seqids'])
            dec_output = self.decoder(enc_output, dec_embed, inputs['seqids'],
                    targets['seqids'])
            # jax.debug.print('enc_input: {}\nenc_output: {}\nenc_mask: {}\n',
                    # enc_input, enc_output, enc_mask)
            # jax.debug.print('dec_input: {}\ndec_output: {}\n',
                    # dec_input, dec_output)

            return dec_output

        B, Ce = enc_input.shape
        _, Cd = dec_input.shape
        jnp.set_printoptions(precision=2, threshold=100000, edgeitems=100, linewidth=180)
        # jax.debug.print('enc_mask: {}\n', enc_mask)

        enc_output = self.encoder(enc_embed, enc_mask)
        def sample_fn(i, dec_input): 
            dec_embed = self.embed_layer(dec_input)
            rng_key = hk.next_rng_key()
            # print(f'{p=}, {enc_output.shape=}, {dec_input.shape=}')
            dec_output = self.decoder(enc_output, dec_embed, enc_mask, dec_mask) / temperature
            dec_step = jax.lax.dynamic_slice(dec_output, (0, i, 0), (B, 1, self.T))
            # print('foo: ', dec_step.shape)
            # jax.debug.print('dec_step: {}', dec_step)
            sample = jax.random.categorical(rng_key, dec_step, axis=2)
            # jax.debug.print('sample: {}', sample)
            # print(f'{dec_step.shape=}, {sample.shape=}, {dec_input.shape=}')
            dec_input = dec_input.at[:,i+1].set(sample[:,0])
            # jax.debug.print('dec_input: {}', dec_input)
            return dec_input
        dec_input = jax.lax.fori_loop(0, Cd-1, sample_fn, dec_input)
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
    def __init__(self, token_histo, bos_id, smoothing_eps=0.1):
        """
        page 8, Label Smoothing: using eps = 0.1
        """
        super().__init__()
        # self.eps = smoothing_eps
        self.bos_id = bos_id
        token_histo = token_histo.astype(jnp.float32)
        self.token_dist = token_histo / jnp.sum(token_histo)
        self.T = self.token_dist.shape[0]
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

        # smoothing
        dist = jnp.expand_dims(self.token_dist, (0,1))
        targets = jax.nn.one_hot(targets, self.T, axis=2)
        targets = (1.0 - self.eps) * targets + self.eps * dist 
        # jax.debug.print('{}', dec_mask)
        dec_pred_logits = dec_output_logits[:,:-1,:]
        dec_pred = jax.nn.softmax(dec_pred_logits, axis=2)
        kldiv = funcs.fused_kldiv_softmax(targets, dec_pred_logits, 2)
        mean_kldiv = kldiv.mean(where=targets_active)
        label_entropy = funcs.entropy(targets, axis=2).mean(where=targets_active)
        model_entropy = funcs.entropy(dec_pred, axis=2).mean(where=targets_active)
        # TODO: don't return model_entropy, it is wrong
        return mean_kldiv, label_entropy, model_entropy


def _wrap_haiku(mod_cls, *args):
    def wrapped_fn(*call_args):
        mod = mod_cls(*args)
        return mod(*call_args)
    return wrapped_fn

def make_model(hps, is_train, token_info):
    n_vocab = token_info['histo'].shape[0]
    mask_id = token_info['mask']
    arch = dict(zip('HMKVFL', (hps.H, hps.M, hps.K, hps.V, hps.F, hps.num_layers)))
    return hk.transform(_wrap_haiku(Model, hps.dropout_rate, hps.pos_encoding_factor,
        arch, is_train, n_vocab, mask_id))

def make_test_model(hps, is_train, token_info):
    n_vocab = token_info['histo'].shape[0]
    return hk.transform(_wrap_haiku(EncoderDecoder, hps, is_train, n_vocab))

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

