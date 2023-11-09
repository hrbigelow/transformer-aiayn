import jax
import jax.numpy as jnp
from jax import custom_vjp

@custom_vjp
def linear(x, w):
    # Position-wise FF 
    return jnp.einsum('bi,ij -> bj', x, w)

def linear_fwd(x, w):
    return linear(x, w), (x, w, jnp.einsum('bi->b', x**2))

def linear_bwd(res, g):
    x, w, norm = res
    xg = jnp.einsum('bj,ij -> bi', g, w)
    wg = jnp.einsum('bj,bi -> bij', g, x)
    return xg, wg / norm[:,None,None]

linear.defvjp(linear_fwd, linear_bwd)

B,I,J = 7, 100, 50
x = jax.random.uniform(jax.random.key(0), (B,I))
w = jax.random.normal(jax.random.key(0), (I,J))

def final(x, w):
    a = linear(x, w)
    return a.mean()

final_grad = jax.grad(final)

final_grad(x, w)



