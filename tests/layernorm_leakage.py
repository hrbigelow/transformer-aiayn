import haiku as hk
import jax
import jax.numpy as jnp

batch, other = 3, 7
rng_key = jax.random.PRNGKey(42)
input = jax.random.normal(rng_key, (batch, other))
out_grads = jnp.zeros((batch, other))
out_grads = out_grads.at[:,4].set(1.0)

def make_grads(axis, input, out_grads):
    def wrap_fn(*args):
        return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True)(*args)

    rng_key = jax.random.PRNGKey(42)
    layer = hk.transform(wrap_fn)
    params = layer.init(rng_key, input)
    primal, vjp_fn = jax.vjp(layer.apply, params, rng_key, input)
    param_grad, rng_grad, input_grad = vjp_fn(out_grads)
    jnp.set_printoptions(precision=2, linewidth=150)
    print(input_grad)

print('input gradient for layer norm with axis=(0,)')
make_grads((0,), input, out_grads)

print('\n\ninput gradient for layer norm with axis=(0,1)')
make_grads((0,1), input, out_grads)

"""
input gradient for layer norm with axis=(0,)
[[ 0.00e+00  0.00e+00  0.00e+00  0.00e+00  9.83e-08  0.00e+00  0.00e+00]
 [ 0.00e+00  0.00e+00  0.00e+00  0.00e+00 -7.06e-08  0.00e+00  0.00e+00]
 [ 0.00e+00  0.00e+00  0.00e+00  0.00e+00 -2.77e-08  0.00e+00  0.00e+00]]

input gradient for layer norm with axis=(0,1)
[[-0.09 -0.21 -0.17 -0.17  1.03 -0.21 -0.17]
 [-0.15 -0.24 -0.15 -0.19  0.98 -0.11 -0.17]
 [-0.17 -0.11 -0.19 -0.14  0.99 -0.2  -0.15]]
"""

