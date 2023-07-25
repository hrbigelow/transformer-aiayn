import fire
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from aiayn import model, data, hparams
import pdb


def main(query, ckpt_dir, resume_ckpt, tokenizer_name, token_info_file, 
        hps_keys: str = 'arch,reg,data,sample', 
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

    mod = model.make_model(hps, False, token_info) 
    mngr = orbax.CheckpointManager(
        hps.ckpt_dir, orbax.Checkpointer(orbax.PyTreeCheckpointHandler()))
    params = mngr.restore(hps.resume_ckpt)
    print('Restored model from checkpoint')

    rng_key = jax.random.PRNGKey(hps.random_seed)
    print('Got random key')
    query_toks = data.tokenize(query)

    print('Received query:')
    print(data.de_tokenize(np.expand_dims(query_toks, 0))[0])
    bos_id = token_info['bos'].item()
    special_toks = data.get_special_tokens(token_info)

    def tokids(shape): 
        return jnp.reshape(jnp.tile(jnp.arange(shape[1]), shape[0]), shape) 

    query_toks = jnp.repeat(jnp.expand_dims(query_toks, axis=0), hps.num_sample, axis=0)
    query_mask = jnp.zeros_like(query_toks, dtype=jnp.int32)
    query_tokids = tokids(query_toks.shape)
    dec_input = jnp.full((hps.num_sample, hps.max_target_len), bos_id, dtype=np.int32)
    dec_seqids = jnp.zeros_like(dec_input, dtype=jnp.int32)
    dec_tokids = tokids(dec_input.shape)

    inputs = dict(seqs=query_toks, seqids=query_mask, tokids=query_tokids)
    targets = dict(seqs=dec_input, seqids=dec_seqids, tokids=dec_tokids)

    # print(f'{query_toks.shape=}, {dec_input.shape=}')
    pred_toks = mod.apply(params, rng_key, inputs, targets, hps.temperature) 
    print(pred_toks)
    pred_sentences = data.de_tokenize(pred_toks, special_toks)
    print('\n'.join(pred_sentences))


if __name__ == '__main__':
    fire.Fire(main)
