import fire
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from aiayn import model, data, hparams
import pdb


def main(query, ckpt_dir, resume_ckpt, tokenizer_name, token_info_file,
        print_special_toks, hps_keys: str = 'arch,reg,data,sample', 
        **hps_overrides
        ):

    opts = { 
            'ckpt_dir': ckpt_dir, 
            'resume_ckpt': resume_ckpt, 
            'token_info_file': token_info_file, 
            **hps_overrides 
            }
    hps = hparams.setup_hparams(hps_keys, opts)
    data.set_config(data_dir=hps.data_dir, tokenizer=tokenizer_name)
            
    print('Running with parameters:')
    print(hps)

    token_info = data.load_token_info(hps.token_info_file)
    n_vocab = token_info['histo'].shape[0]
    mask_id = token_info['mask']
    bos_id = token_info['bos']

    mod = model.make_test_model(hps, n_vocab, bos_id, mask_id) 
    mngr = orbax.CheckpointManager(
        hps.ckpt_dir, orbax.Checkpointer(orbax.PyTreeCheckpointHandler()))
    params = mngr.restore(hps.resume_ckpt)
    print('Restored model from checkpoint')

    rng_key = jax.random.PRNGKey(hps.random_seed)
    print('Got random key')
    query_toks = data.tokenize(query)
    query_toks = jnp.repeat(query_toks[None,:], hps.num_sample, axis=0)

    print('Received query:')
    print(data.de_tokenize(query_toks[0:1,:])[0])
    bos_id = token_info['bos'].item()

    if print_special_toks:
        special_toks = data.get_special_tokens(token_info)
    else:
        special_toks = {}

    pred_toks = mod.apply(params, rng_key, query_toks, hps.max_target_len, hps.temperature)
    print(pred_toks)
    pred_sentences = data.de_tokenize(pred_toks, special_toks)
    print('\n'.join(pred_sentences))


if __name__ == '__main__':
    fire.Fire(main)
