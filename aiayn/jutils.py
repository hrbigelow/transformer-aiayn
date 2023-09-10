import jax
import jax.numpy as jnp

def tree_add(accum, val):
    # tree_map fn over leaves of accum and val
    # if accum is None, instead feed 'None' as a leaf
    if accum is None:
        accum = jax.tree_map(jnp.zeros_like, val)
    return jax.tree_map(jax.lax.add, accum, val)

def map_sum(map_fn, data, rng):
    """
    Map map_fn across the items of data (split by the leading axis)
    Returns: sum of mapped items, new_rng
    """
    initial_data = jax.tree_map(lambda x: x[0], data)
    rest_of_data = jax.tree_map(lambda x: x[1:], data)
    result = map_fn(initial_data, rng)
    rng, = jax.random.split(rng, 1)
    carry = result, rng

    def scan_fn(carry, item):
        accu, rng = carry
        result = map_fn(item, rng)
        rng, = jax.random.split(rng, 1)
        accu = jax.tree_map(lambda x, y: x + y, accu, result)
        carry = accu, rng
        return carry, 0
    carry, _ = jax.lax.scan(scan_fn, carry, data) 
    return carry

