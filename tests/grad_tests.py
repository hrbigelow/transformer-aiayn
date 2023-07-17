import fire
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from aiayn import model, hparams


arch = dict(zip('HMKVFL', (2, 16, 23, 32, 17, 3)))
dropout_rate = 0.1

def make_mask(B, C):
    # Create a generic mask with the last half masked
    return jnp.indices((B,C))[1] * 2 // C

def print_grad(grad):
    jnp.set_printoptions(precision=2, threshold=100000, edgeitems=100, linewidth=180)
    print(grad, '\n\n')

def ff():
    H, M, K, V, F, L = tuple(arch[l] for l in 'HMKVFL')
    B, Q = 2, 13
    rng_key = jax.random.PRNGKey(42)
    input = jax.random.normal(rng_key, (B,Q,M))
    mask = jnp.array([ [0] * 8 + [1] * 5, [0] * 6 + [1] * 7 ])
    grads = model.make_grads(model.PositionwiseFF, (M,F), (B,Q,M), (input, mask)) 
    _, _, input_grad, _ = grads
    print_grad(input_grad)

def attn(use_qmask=False, use_tmask=False):
    B, Q, T = 2, 13, 13
    H, M, K, V, F, L = tuple(arch[l] for l in 'HMKVFL')
    rng_key = jax.random.PRNGKey(42)
    kvinput = jax.random.normal(rng_key, (B,T,M))
    qinput = jax.random.normal(rng_key, (B,T,M))

    qmask = make_mask(B,Q) if use_qmask else None
    tmask = make_mask(B,T) if use_tmask else None

    call_args = kvinput, qinput, qmask, tmask, None
    grads = model.make_grads(model.MultiHeadAttention, (H, M, K, V), (B,Q,M), call_args)
    _, _, kv_grad, q_grad, _, _, _ = grads
    print('kv_grad:')
    print_grad(kv_grad)

    print('\n\nq_grad:')
    print_grad(q_grad)

def encoder():
    B, Q, T, M = 2, 13, 13, arch['M']
    rng_key = jax.random.PRNGKey(42)
    input = jax.random.normal(rng_key, (B,T,M))
    mask = jnp.array([ [0] * 8 + [1] * 5, [0] * 6 + [1] * 7 ])
    inst_args = 0.2, arch, True 
    call_args = input, mask
    grads = model.make_grads(model.Encoder, inst_args, (B,Q,M), call_args) 
    _, _, input_grad, _ = grads
    print_grad(input_grad)

def decoder():
    B, Cd, Ce, M = 2, 23, 13, arch['M']
    T = 10
    rng_key = jax.random.PRNGKey(42)
    enc_out = jax.random.normal(rng_key, (B,Ce,M))
    dec_in = jax.random.normal(rng_key, (B,Cd,M))
    enc_mask = make_mask(B,Ce)
    dec_mask = make_mask(B,Cd)

    inst_args = 0.1, arch, T, True, None
    decoder = hk.transform(model._wrap_haiku(model.Decoder, *inst_args))
    call_args = enc_out, dec_in, enc_mask, dec_mask
    params = decoder.init(rng_key, *call_args)
    primal, vjp_fn = jax.vjp(decoder.apply, params, rng_key, *call_args)
    out_grad = jnp.zeros((B,Cd,T), dtype=jnp.float32)
    out_grad = out_grad.at[:,5,:].set(1.0)
    print(out_grad)
    grads = vjp_fn(out_grad)
    _, _, _, dec_in_grad, _, _ = grads
    print_grad(dec_in_grad)

    # grads = model.make_grads(model.Decoder, inst_args, (B,Q,M), call_args) 

def decoder_layer():
    B, Cd, Ce, M = 2, 23, 13, arch['M']
    T = 10
    rng_key = jax.random.PRNGKey(42)
    enc_out = jax.random.normal(rng_key, (B,Ce,M))
    dec_in = jax.random.normal(rng_key, (B,Cd,M))
    enc_mask = make_mask(B,Ce)
    dec_mask = make_mask(B,Cd)
    inst_args = 0.1, arch, True, 0
    out_grad = jnp.zeros((B,Cd,M), dtype=jnp.float32)
    out_grad = out_grad.at[:,5,:].set(1.0)
    call_args = enc_out, dec_in, enc_mask, dec_mask
    grads = model.make_grads(model.DecoderLayer, inst_args, out_grad, call_args)
    _, _, _, dec_in_grad, _, _ = grads
    print_grad(dec_in_grad)


def model_grads():
    H, M, K, V, F, L = tuple(arch[l] for l in 'HMKVFL')
    B, Q, T = 2, 13, 20
    inst_args = dropout_rate, arch, True  
    out_shape = (B, Q, T)
    input = jax.random.normal(rng_key, (B,T,M))
    mask = jnp.array([ [0] * 8 + [1] * 5, [0] * 6 + [1] * 7 ])
    call_args = enc_input, dec_input

    # grads = model.make_grads(model.Model, inst_args, 

if __name__ == '__main__':
    fire.Fire(dict(ff=ff, attn=attn, encoder=encoder, decoder=decoder,
        decoder_layer=decoder_layer))

