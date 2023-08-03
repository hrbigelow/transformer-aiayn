import fire
import sys
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from aiayn import model, data, hparams
import pdb


def main(ckpt_dir, resume_ckpt, tokenizer_name, token_info_file,
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

    tok_map = data.load_token_info(hps.token_info_file)

    mod = model.make_test_model(hps, tok_map) 
    mngr = orbax.CheckpointManager(
        hps.ckpt_dir, orbax.Checkpointer(orbax.PyTreeCheckpointHandler()))
    state = mngr.restore(hps.resume_ckpt)
    params = state['params']
    print('Restored model from checkpoint')

    rng_key = jax.random.PRNGKey(hps.random_seed)

    if print_special_toks:
        special_toks = data.get_special_tokens(token_info)
    else:
        special_toks = {}
    print('Enter text (Ctrl-D to exit)')

    try:
        while True:
            query = input('> ')
            if len(query) == 0:
                continue
            query_toks = data.tokenize(query)
            query_toks = jnp.repeat(query_toks[None,:], hps.num_sample, axis=0)
            pred_toks = mod.apply(params, rng_key, query_toks, hps.beam_search_alpha,
                    hps.beam_search_beta, hps.beam_size, hps.max_target_len)
            # print(pred_toks)
            print('Inference for batch element 0:')
            pred_sentences = data.de_tokenize(pred_toks[0], tok_map['eos'], special_toks)
            print('\n'.join(pred_sentences))
            print('\n')
    except EOFError:
        print('Bye')
        sys.exit(0)


if __name__ == '__main__':
    fire.Fire(main)
