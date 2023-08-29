import jax
import jax.numpy as jnp
from aiayn import funcs

def fn(x, w):
    s = jax.nn.softmax(x, axis=0, where=w, initial=0.0)
    return jnp.sum(s, where=w, initial=0.0)

def lfn(x, w):
    s = jax.nn.log_softmax(x, axis=0, where=w, initial=0.0)
    return jnp.sum(s, where=w, initial=0.0)

def my_fn(x, w):
    s = funcs.softmax(x, axis=0, where=w, initial=0.0)
    return jnp.sum(s, where=w, initial=0.0)

def my_lfn(x, w):
    s = funcs.log_softmax(x, axis=0, where=w, initial=0.0)
    return jnp.sum(s, where=w, initial=0.0)

x = jnp.array([1.0, 1.0, 100.0]) 
w = jnp.array([True, True, False])

print('fn, grad of fn:', fn(x, w), jax.grad(fn)(x, w))
print('lfn, grad of lfn:', lfn(x, w), jax.grad(lfn)(x, w))
print('my_fn(x, None), my_fn(x, w), grad of my_fn:', my_fn(x, None), my_fn(x, w), jax.grad(my_fn)(x, w))
print('my_lfn, grad of my_lfn:', my_lfn(x, w), jax.grad(my_lfn)(x, w))


print(f'{funcs.log_softmax(x, axis=0)=}')
print(f'{funcs.log_softmax(x, axis=0, where=w, initial=0.0)=}')


