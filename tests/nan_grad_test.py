import jax
import jax.numpy as jnp
from aiayn import funcs

if __name__ == '__main__':
    grad_fn = jax.grad(funcs.entropy)
    x = jnp.array([0.5, 0.5, 0.0])
    w = jnp.array([True, True, False])
    print(f'grad_fn({x=}, {w=}) = ', grad_fn(x, 0, w))
    
    w2 = jnp.array([True, True, True])
    print(f'grad_fn({x=}, {w=}) = ', grad_fn(x, 0, w2))

    x2 = jnp.array([0.33, 0.33, 0.33]) 
    print(f'grad_fn({x=}, {w=}) = ', grad_fn(x2, 0, w))

"""
Prints:
grad_fn(x=Array([1., 0.], dtype=float32), w=Array([ True, False], dtype=bool)) =  [ 1. nan]
grad_fn(x=Array([1., 0.], dtype=float32), w=Array([ True, False], dtype=bool)) =  [1. 0.]
"""
