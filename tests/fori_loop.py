import jax
import jax.numpy as jnp

def accu(x, steps):
    def add_one(i, val):
        if val is None:
            return x
        else:
            return val + 1.0

    return jax.lax.fori_loop(0, steps, add_one, None)

accu_repl = jax.pmap(accu) 


x = jax.random.normal(jax.random.PRNGKey(42), (5,3))

y = accu_repl(x, 6)


