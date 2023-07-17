import sys
import jax
import jax.numpy as jnp
import numpy as np
from aiayn import hparams, model, data

def main(data_path):
    hps_keys = 'arch,reg,train,data,logging'
    hps = hparams.setup_hparams(hps_keys, dict(data_path=data_path))
    token_info = data.load_token_info(hps.data_path) 
    mod = model.make_model(hps, True, token_info)

    B, C = 16, 100
    enc_input = np.empty((B, C), dtype=np.uint16)
    dec_input = np.empty((B, C), dtype=np.uint16)
    rng_key = jax.random.PRNGKey(42)

    params = mod.init(rng_key, enc_input, dec_input)
    print(type(params))
    for k, v in params.items():
        print(f'{k}: {v.keys()}')
    # _, key_paths = jax.tree_util.tree_flatten_with_path(params)
    # print(key_paths)
    # leaves = jax.tree_util.tree_leaves(params)
    # print(leaves)
    tree_def = jax.tree_util.tree_structure(params)
    # print(tree_def)

if __name__ == '__main__':
    data_path = sys.argv[1]
    main(data_path)


