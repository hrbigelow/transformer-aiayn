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
    Maps map_fn across the items of data (split by the leading axis)
    Returns: sum of mapped items, new_rng

    Inputs:
        data: a pytree of tensor leaves.  Each tensor has the same sized axis 0
        rng:  random seed
        map_fn:  (data_slice, rng) -> result
                 where data_slice is a pytree of one slice along axis 0 of each
                 leaf of `data`
    Returns:
        sum of each result returned by map_fn 
    """
    initial_data = jax.tree_map(lambda x: x[0], data)
    result_shape = jax.eval_shape(map_fn, initial_data, rng)
    result = jax.tree_map(lambda dst: jnp.zeros(dst.shape, dst.dtype), result_shape)
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


