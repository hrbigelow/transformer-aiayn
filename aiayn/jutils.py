import jax
import jax.numpy as jnp

def tree_add(accum, val):
    # tree_map fn over leaves of accum and val
    # if accum is None, instead feed 'None' as a leaf
    if accum is None:
        accum = jax.tree_map(jnp.zeros_like, val)
    return jax.tree_map(jax.lax.add, accum, val)

