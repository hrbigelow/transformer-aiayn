import tensorflow as tf
import jax
import jax.numpy as jnp
import fire
import flax
from tokenizers import Tokenizer
import optax
from aiayn import pack, model, data, hparams, train, utils

from orbax.checkpoint import (AsyncCheckpointer, CheckpointManager,
        CheckpointManagerOptions, SaveArgs, PyTreeCheckpointHandler)

def main(hps_keys: str = 'arch,reg,train,data,logging', **hps_overrides):
    hps = hparams.setup_hparams(hps_keys, hps_overrides)
    num_replicas = jax.local_device_count()
    repl_batch_size = hps.batch_dim0 // num_replicas
    rng_key = jax.random.PRNGKey(42)

    # Initialize tokenizer
    tokenizer = Tokenizer.from_str(tf.io.gfile.GFile(hps.tokenizer_file).read())
    n_vocab = tokenizer.get_vocab_size() + 2 # add BOS and EOS
    bos_id = tokenizer.token_to_id('[BOS]')
    eos_id = tokenizer.token_to_id('[EOS]')
    pad_id = tokenizer.token_to_id('[PAD]')

    # Create model and objective
    mod = model.make_model(hps, bos_id, eos_id, n_vocab, True)
    objective = model.Objective(bos_id, n_vocab, hps.label_smooth_eps, hps.attn_loss_weight)

    lr_fn = train.make_learning_rate_fn(hps.warmup_steps, hps.M)
    tx = optax.adam(learning_rate=lr_fn, b1=hps.adam_beta1, b2=hps.adam_beta2,
            eps=hps.adam_eps)

    # Create update function
    update_fn = train.make_update_fn(mod, objective, repl_batch_size, hps.accum_steps,
            hps.with_metrics, tx, hps.ckpt_dir)

    # Create dataset
    feature_lengths = { 'inputs': hps.max_source_len, 'targets': hps.max_target_len }
    token_ds = data.load_tfrecord_dataset(hps.dataset_glob, hps.swap_source_target)
    token_ds = data.add_special_tokens(token_ds, bos_id, eos_id) 
    token_ds = token_ds.repeat().shuffle(hps.shuffle_size, rng_key[0], True)
    pack_ds = pack.pack_dataset(token_ds, feature_lengths, 1000, 10, pad_id) 
    dataset = pack_ds.rebatch(hps.batch_dim0)

    # create args for restoring checkpoint
    item = next(dataset.as_numpy_iterator())
    item = jax.tree_map(lambda ten: ten[:1], item)
    params = mod.init(rng_key, item['inputs'], item['targets'])
    opt_state = tx.init(params)

    pre_init_state = dict(params=params, opt_state=opt_state)
    shardings = jax.tree_map(lambda x: x.sharding, pre_init_state)
    init_state = dict(state=pre_init_state, item=item)
    restore_args = utils.construct_restore_args(init_state, dict(state=shardings,
        item=item))

    # Restore from checkpoint
    options = CheckpointManagerOptions(save_interval_steps=hps.ckpt_every, max_to_keep=20)
    checkpointer = AsyncCheckpointer(PyTreeCheckpointHandler(), timeout_secs=100)
    mngr = CheckpointManager(hps.ckpt_dir, checkpointer, options)
    dump = mngr.restore(hps.resume_ckpt, items=init_state, restore_kwargs=
            {'restore_args': restore_args})
    state = dump['state']
    state.update({'rng': rng_key})
    item = dump['item']

    num_replicas = jax.local_device_count()
    batch_repl_size = hps.batch_dim0 // num_replicas
    shape = [num_replicas, batch_repl_size, -1]
    def reshape_fn(ten):
        return jnp.reshape(ten, shape)

    print(f'Restored from checkpoint {hps.resume_ckpt}')


    update_fn_m = jax.pmap(update_fn, axis_name='batch')
    state_m = flax.jax_utils.replicate(state)
    item = jax.tree_map(reshape_fn, item)
    inputs = item['inputs']
    targets = item['targets']
    print(inputs['seqs'].shape)

    # Call the function
    state_m, metrics_m = update_fn_m(state_m, inputs, targets) 

if __name__ == '__main__':
    fire.Fire(main)

