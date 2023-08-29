import fire
import jax
import jax.numpy as jnp
import numpy as np
from aiayn import model, hparams, data

def main(token_info_file, hps_keys: str = 'arch,reg,train,data,logging', **hps_overrides):
    rng_key = jax.random.PRNGKey(42)
    defaults = dict(token_info_file=token_info_file)
    defaults.update(hps_overrides)
    hps = hparams.setup_hparams(hps_keys, defaults)
    tok_map = data.load_token_info(hps.token_info_file)
    mod = model.make_train_model(hps, tok_map)
    dummy = jnp.zeros((1,10), dtype=jnp.int32)
    enc_input = dict(seqs=dummy, seqids=dummy, tokids=dummy, counts=dummy)
    dec_input = dict(seqs=dummy, seqids=dummy, tokids=dummy, counts=dummy)
    params = mod.init(rng_key, enc_input, dec_input)  
    map_fn = lambda path, par: f'{path}: {par.shape}'
    shapes = jax.tree_util.tree_map_with_path(map_fn, params)
    shapes_list, _ = jax.tree_util.tree_flatten(shapes)
    print('\n'.join(shapes_list))
    total_params = jax.tree_util.tree_reduce(lambda tot, par: tot + par.size, params, 0)
    print(f'{total_params=}')

if __name__ == '__main__':
    fire.Fire(main)

