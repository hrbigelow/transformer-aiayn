import jax.numpy as jnp
from jax import custom_jvp, custom_vjp

@custom_jvp
def take(matrix, indices, axis):
    """
    A "differentiable" version of jnp.take.
    gradient w.r.t. matrix is as usual
    gradient w.r.t. indices indices is equivalent to the sum of gradient of the one-hot
    component of a one-hot transformed indicess across all matrix components
    """
    return jnp.take(matrix, indices, axis=axis)

@take.defjvp
def take_jvp(primals, tangents):
    matrix, indices, axis = primals
    matrix_dot, indices_dot = tangents
    primal_out = take(matrix, indices, axis)
    # tangent_out = 






