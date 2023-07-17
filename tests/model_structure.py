import fire
import jax
import jax.numpy as jnp
import numpy as np
from aiayn import model, hparams

def main(token_info_file, hps_keys: str = 'arch,reg,train,data,logging', **hps_overrides):
    rng_key = jax.random.PRNGKey(42)
    defaults = dict(token_info_file=token_info_file)
    defaults.update(hps_overrides)
    hps = hparams.setup_hparams(hps_keys, defaults)
    token_info = np.load(hps.token_info_file)
    mod = model.make_model(hps, True, token_info)
    B = hps.batch_dim0
    C = hps.max_sentence_length
    enc_input = jnp.zeros((B,C), dtype=jnp.int32)
    dec_input = jnp.zeros((B,C), dtype=jnp.int32)
    params = mod.init(rng_key, enc_input, dec_input)  
    map_fn = lambda path, par: f'{path}: {par.shape}'
    shapes = jax.tree_util.tree_map_with_path(map_fn, params)
    shapes_list, _ = jax.tree_util.tree_flatten(shapes)
    print('\n'.join(shapes_list))

if __name__ == '__main__':
    fire.Fire(main)

