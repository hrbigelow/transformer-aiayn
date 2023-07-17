import numpy as np
import haiku as hk
import jax.numpy as jnp
import jax

class Embed(hk.Module):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, x):
        w = hk.get_parameter('w', [10, 20], np.float32, init_fn)
        return jnp.take(w, x, axis=0)

class DeEmbed(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        w = hk.get_global_parameter('global_embed')
        return jnp.take(jnp.transpose(w), x, axis=0)

class Parent(hk.Module):
    def __init__(self):
        super().__init__()
        init_fn = hk.initializers.RandomNormal(1.0, 0.0) 
        # it is an error to call this twice for same global_name
        hk.declare_global_parameter('global_embed', [100, 50], np.float32, init_fn)
        self.embed = Embed()
        self.debed = DeEmbed()

    def __call__(self, x):
        e = self.embed(x)
        d = self.debed(e)
        return d

def make_mod(cls, name):
    def fn(*call_args):
        mod = cls(name)
        return mod(*call_args)
    return fn

def main():
    model = make_mod(Parent)
    key = jax.random.PRNGKey(42)

    params = model.init(key, jnp.empty((5, 100)))
    # params = { 
    # 'Parent': { 
    #   'Embed': { 'global_embed': <Array> }, 
    #   'DeEmbed': { 'global_embed': <Array> },
    # }
    # where <Array> is the same object 


def parameter_shapes(params):
  """Make printing parameters a little more readable."""
  return jax.tree_util.tree_map(lambda p: p.shape, params)

def main():
    parent = hk.transform(make_mod(Parent, 'Main'))
    # mod1 = hk.transform(make_mod(Mod1, 'Floppy'))
    # mod2 = hk.transform(make_mod(Mod2, 'Skippy'))
    key = jax.random.PRNGKey(42)
    param1 = parent.init(key, jnp.empty((5, 100)))
    # par1 = mod1.init(key, jnp.empty((5, 100)))
    # par2 = mod2.init(key, jnp.empty((5, 100)))
    print(parameter_shapes(param1))
    # print(parameter_shapes(par2))

if __name__ == '__main__':
    main()



                
                
