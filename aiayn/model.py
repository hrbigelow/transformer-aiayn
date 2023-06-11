from dataclasses import dataclass
import re
import numpy as np
from . import hparams
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
        super().__init__(name=None)
        self.mscale = np.sqrt(hps.M) ** -1
        self.vscale = np.sqrt(hps.V) ** -1
        self.scale_factor = np.sqrt(hps.K)

    def __call__(self, kvinput, qinput):
        """
        kvinput: bcm 
        qinput: bcm 
        output: bcm 
        """
        kqv_init = hk.initializers.RandomNormal(self.mscale, 0.0)
        o_init = hk.initializers.RandomNormal(self.vscale, 0.0)
        dtype = kvinput.dtype

        wq = hk.get_parameter('wq', [hps.H,hps.M,hps.K], dtype, kqv_init)
        wk = hk.get_parameter('wk', [hps.H,hps.M,hps.K], dtype, kqv_init)
        wv = hk.get_parameter('wv', [hps.H,hps.M,hps.V], dtype, kqv_init)
        wo = hk.get_parameter('wo', [hps.H,hps.V,hps.M], dtype, o_init)

        kval = jnp.einsum('bcm,hmk->bhck', kvinput, wk)
        qval = jnp.einsum('bcm,hmk->bhck', qinput, wq)
        vval = jnp.einsum('bcm,hmv->bhcv', kvinput, wv)
        alogit = jnp.einsum('bhck,bhdk->bhcd', kval, qval)
        if self.masked:
            # query (d) can only attend to keys (c) at or before it 
            upper_tri = (jnp.tri(a.logit.shape[2], k=-1) - 1.0) * 100.0
            alogit += upper_tri 
        att = jax.nn.softmax(alogit * self.scale_factor ** -1, dim=2)
        pre = jnp.einsum('bhcd,bhcv->bhdv', att, vval)
        out = jnp.einsum('bhcv,hvm->bcm', pre, wo)

class PositionwiseFF(hk.Module):
    """
    Implements equation 2 (section 3.3)
    """
    def __init__(self, hps):
        super().__init__()
        self.mscale = np.sqrt(hps.M) ** -1
        self.fscale = np.sqrt(hps.F) ** -1
        self.M = hps.M

    def forward(self, input):
        """
        input: bcm
        """
        dtype = input.dtype
        norm_init = hk.initializers.RandomNormal()
        w1 = hk.get_parameter('w1', [hps.M,hps.F], dtype, norm_init * self.mscale)
        b1 = hk.get_parameter('b1', [hps.F], jnp.zeros)
        w2 = hk.get_parameter('w2', [hps.F,hps.M], dtype, norm_init * fscale)
        b2 = hk.get_parameter('b2', [hps.M], jnp.zeros)
        s = F.relu(t.einsum('bcm,mf->bcf', input, w1) + b1)
        out = t.einsum('bcf,fm->bcm', s, w2) + b2
        return out

class DropoutAddAndNorm(hk.Module):
    def __init__(self, hps):
        super().__init__()
        self.rate = hps.dropout_rate
        self.dropout = hk.dropout(hps.dropout_rate)
        self.layer_norm = hk.LayerNorm(1, create_scale=True, create_offset=True)

    def __call__(self, residual, proximal, is_training):
        """
        residual: bdm
        proximal: bdm
        output: bdm
        """
        if is_training:
            proximal = self.dropout(hk.next_rng_key(), self.rate, proximal)
        return self.layer_norm(residual + proximal)

class InputEmbedding(hk.Module):
    def __init__(self, T, hps):
        super().__init__()
        self.T = T
        self.M = hps.M
        self.scale_factor = np.sqrt(self.T) ** -1 # my experiment
        self.pos_factor = hps.pos_encoding_factor

    def positional_embedding(self, num_positions):
        pos = jnp.arange(num_positions)
        denom = 10000 ** jnp.linspace(0, 1, self.M)
        arg = pos.unsqueeze(-1) / denom.unsqueeze(0)
        # it shouldn't matter what order embeddings are placed but here I follow
        # the paper, and alternate sin with cos
        pos_emb = jnp.empty(num_positions,self.M)
        pos_emb[:,::2] = jnp.sin(arg[:,::2])
        pos_emb[:,1::2] = jnp.cos(arg[:,1::2])
        return pos_emb

    def __call__(self, input):
        """
        input: bc (values: integer-encoded token ID)
        output: bcm
        """
        C = input.shape[1]
        pos_embed = self.positional_embedding(C)
        init = hk.initializers.RandomNormal(1.0, 0.0)
        embed_matrix = hk.get_parameter('emb', [self.T, self.M], init)
        # embed = self.embedding(input) * self.scale_factor
        embed = jnp.take(embed_matrix, input, axis=1)
        return embed + pos_embed * self.pos_factor

class EncoderLayer(hk.Module):
    def __init__(self, hps):
        super().__init__()
        self.att = MultiHeadAttention(hps)
        self.norm1 = DropoutAddAndNorm(hps)
        self.ff = PositionwiseFF(hps)
        self.norm2 = DropoutAddAndNorm(hps)

    def __call__(self, input, is_train):
        """
        input: bcm
        returns: bcm
        """
        att = self.att(input, input)
        norm = self.norm1(input, att, is_train)
        ff = self.ff(norm)
        out = self.norm2(norm, ff, is_train)
        return out

class Encoder(hk.Module):
    def __init__(self, hps, embed_layer):
        super().__init__()
        self.embed_layer = embed_layer 
        mods = (EncoderLayer(hps) for _ in range(hps.num_layers))
        self.body = hk.Sequential(*mods)

    def __call__(self, input, is_train):
        """
        input: bc
        returns: bcm
        """
        out = self.embed_layer(input)
        out = self.body(out)
        return out

class DecoderLayer(hk.Module):
    def __init__(self, hps):
        super().__init__()
        self.mask_att = MultiHeadAttention(hps, masked=True)
        self.norm1 = DropoutAddAndNorm(hps)
        self.att2 = MultiHeadAttention(hps)
        self.norm2 = DropoutAddAndNorm(hps)
        self.ff = PositionwiseFF(hps)
        self.norm3 = DropoutAddAndNorm(hps)

    def __call__(self, enc_out, input, is_train):
        """
        enc_out: bcm
        queries: bcm
        """
        att1 = self.mask_att(input, input)
        norm1 = self.norm1(input, att1, is_train)
        att2 = self.att2(enc_out, norm1)
        norm2 = self.norm2(norm1, att2, is_train)
        ff = self.ff(norm2)
        out = self.norm3(norm2, ff, is_train)
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

    def __call__(self, side_input, input, is_train):
        out = input
        for mod in self.layers:
            out = mod(side_input, out, is_train)
        return out

class Decoder(hk.Module):
    def __init__(self, hps, T, embed_layer):
        super().__init__()
        self.embed_layer = embed_layer 
        self.body = TeeSequential(*(DecoderLayer(hps) for _ in range(hps.num_layers)))
        self.linear_final = jax.nn.Linear(hps.M, T)
        self.softmax = jax.nn.Softmax(dim=2)

    def __call__(self, enc_out, dec_in, is_train):
        """
        enc_out: bcm 
        dec_in: bc
        returns: bct
        """
        out = self.embed_layer(dec_in)
        out = self.body(enc_out, out, is_train)
        out = self.linear_final(out)
        out = self.softmax(out)
        return out

class Embed(hk.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model

    def __call__(self, tokens):
        init = hk.initializers.RandomNormal(1.0, 0.0)
        embed_matrix = hk.get_parameter('emb', [self.T, self.M], init)
        return 

class Model(hk.Module):
    def __init__(self, hps, token_histo, pad_token_id):
        super().__init__()
        self.hps = hps
        self.pad_token_id = pad_token_id 
        T = token_histo.shape[0]

        self.embed_layer = InputEmbedding(T, hps) 
        self.encoder = Encoder(hps, self.embed_layer)
        self.decoder = Decoder(hps, T, self.embed_layer)
        self.loss = Objective(token_histo, self.pad_token_id) 

    def __call__(self, is_train, enc_input, dec_input, rng_key):
        """
        enc_input: bc
        dec_input: bc
        returns: bct
        """
        enc_output = self.encoder(enc_input, is_train)
        dec_output = self.decoder(enc_output, dec_input, is_train)
        return dec_output

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

    def regex_params(self, pat):
        """
        For each parameter name matching pat, return:
        name => (*captures)
        """
        ret = {}
        for name, par in self.params_dict():
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

    def __getstate__(self):
        """
        Provided for pickle since torch.Generator cannot be pickled
        """
        print('in Model::__getstate__')
        d = dict.copy(self.__dict__)
        d['rng'] = self.rng.get_state()
        return d

    def __setstate__(self, state):
        gen = t.Generator()
        gen.set_state(state['rng'])
        state['rng'] = gen
        self.__dict__.update(state)

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
        # self.register_buffer('vocab_size', t.tensor(T), persistent=False)
        self.T = token_histo.shape[0] 

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
        one_hot = jax.nn.one_hot(labels, self.T, -1)
        smoothed = (1.0 - self.eps) * one_hot + self.eps * self.u
        return smoothed

    def __call__(self, dec_input, dec_output):
        """
        dec_input: bc
        dec_output: bct
        """
        # bc
        labels = dec_input[:,1:]
        smoothed_labels = self.label_smooth(labels)
        dec_mask = jnp.not_equal(labels, self.pad_value).astype(jnp.float64)
        # dec_mask = t.ne(labels, self.pad_value).to(t.float64)

        # bct
        dec_pred = dec_output[:,:-1,:]
        kldiv = self.kldiv(smoothed_labels, dec_pred, 2)
        masked = kldiv * dec_mask
        total_targets = dec_mask.sum()
        # target_fraction = total_targets / dec_mask.numel()
        # print(f'{total_targets=}, {target_fraction=}')
        loss = masked.sum() / total_targets
        # print(f'{loss=}')
        return loss

def _wrap_haiku(mod_cls, *args):
    def wrapped_fn(*call_args):
        mod = mod_cls(*args)
        return mod(call_args)
    return wrapped_fn

def make_model(hps, token_info):
    return hk.transform(_wrap_haiku(Model, hps, token_info['de'],
        token_info['pad_token_id']))

def make_objective(token_info):
    return hk.transform(_wrap_haiku(Objective, token_info['de'],
        token_info['pad_token_id']))

