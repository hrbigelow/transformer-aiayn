import fire
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from aiayn import model, data, hparams
import pdb


def main(query, hps_keys: str = 'arch,reg,data,sample', **hps_overrides):

    hps = hparams.setup_hparams(hps_keys, hps_overrides)
    print('Running with parameters:')
    print(hps)

    if hps.ckpt_dir is None:
        raise RuntimeError(f'Must provide the --ckpt_dir option')
    if hps.resume_ckpt is None:
        raise RuntimeError(f'Must provide the --resume_ckpt option')
    if hps.data_path is None:
        raise RuntimeError(f'Must provide the --data_path option')

    token_info = data.load_token_info(hps.data_path) 
    mod = model.make_model(hps, False, token_info=token_info) 
    mngr = orbax.CheckpointManager(
        hps.ckpt_dir, orbax.Checkpointer(orbax.PyTreeCheckpointHandler()))
    params = mngr.restore(hps.resume_ckpt)

    rng_key = jax.random.PRNGKey(hps.random_seed)
    tokenizer = data.get_tokenizer()
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    query_toks = jnp.array([bos_id] + tokenizer(query)['input_ids'] + [eos_id])
    print('Received query:')
    print(data.de_tokenize(np.expand_dims(query_toks, 0))[0])

    query_toks = jnp.repeat(jnp.expand_dims(query_toks, axis=0), hps.num_sample, axis=0)
    dec_input = jnp.empty((hps.num_sample, hps.max_sentence_length), dtype=np.int32)
    dec_input = query_toks.at[:,0].set(bos_id)
    pred_toks = mod.apply(params, rng_key, query_toks, dec_input) 
    pred_sentences = data.de_tokenize(pred_toks)
    print('\n'.join(pred_sentences))


if __name__ == '__main__':
    fire.Fire(main)
