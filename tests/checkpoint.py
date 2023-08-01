import fire
import jax
import jax.numpy as jnp
import optax
from aiayn import model, hparams, utils
import pdb
import orbax
from orbax.checkpoint import (CheckpointManager, CheckpointManagerOptions, 
        SaveArgs, Checkpointer, PyTreeCheckpointer, PyTreeCheckpointHandler)
from orbax.checkpoint import type_handlers

def main(ckpt_dir, step):
    hps = hparams.setup_hparams('tiny,reg,train,data,logging', {})
    rng_key = jax.random.PRNGKey(42)
    tok_map = dict(bos=3000, eos=3001, mask=3002, n_vocab=3003)
    mod = model.make_train_model(hps, tok_map)
    z = jnp.zeros((1, 100), dtype=jnp.int32)
    c = jnp.zeros((1, 10), dtype=jnp.int32)
    inputs = dict(seqs=z, seqids=z, tokids=z, counts=c)
    targets = dict(seqs=z, seqids=z, tokids=z, counts=c)
    params = mod.init(rng_key, inputs, targets) 
    opt = optax.adam(lambda ct: 1e-3, b1=0.99, b2=0.98, eps=0.01) 
    opt_state = opt.init(params)

    # The structure we want to save/restore
    save_state = dict(params=params, opt_state=opt_state)

    shardings = jax.tree_map(lambda x: x.sharding, save_state)
    restore_args = utils.construct_restore_args(save_state, shardings)
    
    # print('automatically constructed restore_args:\n', restore_args)
    # restore_args = jax.tree_map(
            # lambda x: 

    state_save_args = jax.tree_map(lambda _: SaveArgs(aggregate=True), save_state)
    # print('restore_args: ', restore_args)
    # mngr = CheckpointManager(ckpt_dir, Checkpointer(PyTreeCheckpointHandler()))
    mngr = CheckpointManager(ckpt_dir, PyTreeCheckpointer())
    mngr.save(step, save_state, save_kwargs={'save_args': state_save_args})
    # mngr.save(step, save_state)
    # restore_args = jax.tree_util.tree_map(lambda item:
            # type_handlers.RestoreArgs(restore_type=type(item)), save_state)

    restored_state = mngr.restore(step, items=save_state,
            restore_kwargs={'restore_args':restore_args})
    print('Restored structure:')
    print(jax.tree_util.tree_structure(restored_state))
    # print('restored_state: ', restored_state)


if __name__ == '__main__':
    fire.Fire(main)



