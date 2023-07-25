import functools
import psutil
import jax
import jax.numpy as jnp
import flax
import orbax.checkpoint
from orbax.checkpoint import (CheckpointManager, CheckpointManagerOptions, 
        SaveArgs, PyTreeCheckpointer)
import optax
import haiku as hk
import os
import signal
import queue
import time
import sys
import fire
import numpy as np
from aiayn import model, data, hparams, report, funcs, pack

def print_range(pfx, tree):
    def fn(acc, x):
        return jnp.minimum(jnp.min(x), acc[0]), jnp.maximum(jnp.max(x), acc[1])
    amin = jnp.array(float('inf'))
    amax = jnp.array(-float('inf'))
    tmin, tmax = jax.tree_util.tree_reduce(fn, tree, (amin, amax))
    jax.debug.print(pfx + ': {}, {}', tmin, tmax)

def make_learning_rate_fn(warmup_steps, M):
    # from section 5.3, page 7, equation 3
    def lr_fn(step):
        factor = jax.lax.min(step ** -0.5, step * warmup_steps ** -1.5)
        new_lr = M ** -0.5 * factor
        # jax.debug.print('learn_rate: {}', new_lr)
        return new_lr
    return lr_fn

def accumulate_gradient(loss_and_grad_fn, step_size, accum_steps, params, inputs, targets):
    """
    Inspired by https://github.com/google-research/vision_transformer/blob/
         62a446f1b3bb9e470db5689bfd7407a8d91bae8a/vit_jax/utils.py#L99
         accumulate gradients to save on memory
    loss_and_grad_fn: function transformed with jax.pmap, jax.loss_and_grad
    r:  replica index
    enc_input: rbcm
    dec_input: rbdm
    Requires: b % accum_steps == 0
    Returns: tuple of loss, grad
    loss: rb
    grad: pytree of params gradients
    """
    def get_slice(i, tensor):
        return jax.lax.dynamic_slice(tensor, (i * step_size, 0),
                (step_size,) + tensor.shape[1:])

    def map_add(a, b):
        return jax.tree_map(lambda x, y: x + y, a, b)
    
    input_slices = jax.tree_map(functools.partial(get_slice, 0), inputs)
    target_slices = jax.tree_map(functools.partial(get_slice, 0), targets)
    (l, e), g = loss_and_grad_fn(params, input_slices, target_slices)

    def acc_loss_and_grad(i, tup):
        input_slices = jax.tree_map(functools.partial(get_slice, i), inputs)
        target_slices = jax.tree_map(functools.partial(get_slice, i), targets)
        (li, ei), gi = loss_and_grad_fn(params, input_slices, target_slices)
        l, e, g = tup
        return (l + li, map_add(e, ei), map_add(g, gi))

    # 1 accum:  6 sec / 10 steps
    # 5 accum: 9 sec / 10 steps
    # 9 accum: 14 sec / 10 steps 
    l, e, g = jax.lax.fori_loop(1, accum_steps, acc_loss_and_grad, (l, e, g))
    return jax.tree_map(lambda x: x / accum_steps, (l, e, g))

def make_update_fn(model, objective, repl_batch_size, accum_steps, with_metrics, tx):
    step_size = repl_batch_size // accum_steps

    def update_fn(state, inputs, targets):
        """
        The function to pass into jax.pmap
        Returns updated state, rng
        """
        """
        Bind the rng key to the device id (which is unique across hosts)
        Note: This is only used for multi-host training (i.e. multiple computers
        each with multiple accelerators).
        """
        params, opt_state, rng_key = state['params'], state['opt'], state['rng'] 
        _, new_rng_key = jax.random.split(rng_key)
        dropout_rng = jax.random.fold_in(rng_key, jax.lax.axis_index('batch'))

        def loss_fn(params, inputs, targets):
            dec_input = targets['seqs']
            dec_output = model.apply(params, dropout_rng, inputs, targets)
            # print_range('dec_output', dec_output)
            loss, label_ent, model_ent = objective.apply(None, None, dec_input, dec_output)
            return loss, (label_ent, model_ent)

        lg_fn = jax.value_and_grad(loss_fn, has_aux=True)

        # setting accum_steps to 1 results in 
        l, e, g = accumulate_gradient(lg_fn, step_size, accum_steps, params, inputs, targets)
        # averages l and g across replicas, then broadcasts the average
        loss, entropy, grad = jax.lax.pmean((l, e, g), axis_name='batch')
        # loss = jax.lax.pmean(l, axis_name='batch')
        # entropy = jax.lax.pmean(e, axis_name='batch')
        # grad = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)

        # tx.update takes ~6 seconds / 10 steps
        updates, opt_state = tx.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        def tensor_norm(*xs):
            names = ('grad', 'param', 'update')
            return {n:jnp.mean(jax.lax.pow(x, 2.0)) for n,x in zip(names, xs)}

        if with_metrics:
            norms = jax.tree_map(tensor_norm, grad, params, updates)
            state = dict(params=params, opt=opt_state, rng=new_rng_key)
            metrics = dict(loss=loss, entropy=entropy, norms=norms)
            return state, metrics
        else:
            state = dict(params=params, opt=opt_state, rng=new_rng_key)
            metrics = dict(loss=loss, entropy=entropy)
            return state, metrics
            
    return update_fn

def make_line_plot(logger, label, steps, vals, palette='Viridis256'):
    """
    steps: [num_steps]
    vals: [num_steps] or [num_steps, num_lines] 
    Create a tandem_lines plot
    """
    if len(vals.shape) == 1:
        vals = jnp.expand_dims(vals, axis=1)

    vals = jnp.transpose(vals, (1, 0))
    num_lines = vals.shape[0]
    val_steps = jnp.repeat(jnp.expand_dims(steps, axis=0), num_lines, axis=0)
    plot = jnp.stack((val_steps, vals), axis=2)
    logger.tandem_lines(label, plot, palette)

def log_steps(logger, steps, learn_rate, losses, entropies):
    make_line_plot(logger, 'loss', steps, losses)
    make_line_plot(logger, 'learn_rate', steps, learn_rate)
    make_line_plot(logger, 'cond_entropy_bits', steps, entropies, 'RdYlGn8')
    make_line_plot(logger, 'cond_perplexity', steps, jnp.power(2.0, entropies), 'RdYlGn8')

def log_metrics(logger, steps, norms): 

    def svlog(prefix, label, key, values):
        vals = report.get_layer_values(pfx, values, key)
        make_plot_data(logger, label, steps, vals)
   
    cats = {
            'enc_att': 'tx/~/enc/~/(layer\d+)/~/att',
            'dec_att': 'tx/~/dec/~/(layer\d+)/~/att',
            'enc_lnorm': 'tx/~/enc/~/(layer\d+)/~/(res.+)/~/lnorm',
            'dec_lnorm': 'tx/~/dec/~/(layer\d+)/~/(res.+)/~/lnorm',
            }

    param_map = {
            'enc_att': ('wq', 'wo', 'wk', 'wv'),
            'dec_att': ('wq', 'wo', 'wk', 'wv'),
            'enc_lnorm': ('scale', 'offset'),
            'dec_lnorm': ('scale', 'offset'),
            }

    for cat, pfx in cats.items():
        for label, val in norms.items():
            for key in param_map[cat]:
                plot_name = f'{cat}_{label}_{key}'
                svlog(pfx, plot_name, key, val)

def setup_train(hps, rng_key):
    num_replicas = jax.local_device_count()
    if hps.batch_dim0 % num_replicas != 0:
        raise RuntimeError(f'{hps.batch_dim0=} not divisible by {num_replicas=}')
    repl_batch_size = hps.batch_dim0 // num_replicas
    if repl_batch_size % hps.accum_steps != 0:
        raise RuntimeError(f'{repl_batch_size=} not divisible by {hps.accum_steps=}')

    data.set_config(data_dir=hps.data_dir)
    token_info = data.load_token_info(hps.token_info_file)
    # special_toks = data.get_special_tokens(token_info)

    options = CheckpointManagerOptions(save_interval_steps=hps.ckpt_every, max_to_keep=10)
    checkpointer = PyTreeCheckpointer()
    mngr = CheckpointManager(hps.ckpt_dir, checkpointer, options)

    lr_fn = make_learning_rate_fn(hps.warmup_steps, hps.M)
    tx = optax.chain(
            optax.adam(learning_rate=lr_fn, b1=hps.adam_beta1, b2=hps.adam_beta2,
                eps=hps.adam_eps)
            )

    mod = model.make_model(hps, True, token_info)
    objective = model.make_objective(hps, token_info)

    update_fn = make_update_fn(mod, objective, repl_batch_size, hps.accum_steps,
            hps.with_metrics, tx)

    initial_step = int(hps.resume_ckpt or 0)

    feature_lengths = { 'inputs': hps.max_source_len, 'targets': hps.max_target_len }
    token_ds = data.load_tfrecord_dataset(hps.dataset_glob, hps.swap_source_target)
    token_ds = token_ds.repeat().shuffle(hps.shuffle_size, rng_key[0], True)
    pack_ds = pack.pack_dataset(token_ds, feature_lengths, 1000, 10, -1) 
    dataset = pack_ds.rebatch(hps.batch_dim0)

    if hps.resume_ckpt:
        params = mngr.restore(hps.resume_ckpt)
        opt_state = tx.init(params)
        initial_step = hps.resume_ckpt
    else:
        item = next(dataset.as_numpy_iterator())
        item = jax.tree_map(lambda ten: ten[:1], item)
        params = mod.init(rng_key, item['inputs'], item['targets'])
        opt_state = tx.init(params)
        initial_step = 0

    return update_fn, dataset, params, opt_state, initial_step, mngr, lr_fn

def train_loop(hps, update_fn, learn_rate_fn, dataset, params, opt_state, mngr,
        initial_step, rng_key, logger):
    num_replicas = jax.local_device_count()
    batch_repl_size = hps.batch_dim0 // num_replicas
    shape = [num_replicas, batch_repl_size, -1]

    if hps.with_metrics: 
        names = ('grad', 'param', 'update')
        fn = lambda x: { l: np.empty(hps.report_every) for l in names } 
        norms = jax.tree_map(fn, params)

    steps = np.empty(hps.report_every)
    losses = np.empty(hps.report_every)
    entropies = np.empty((hps.report_every, 2)) # data, model
    learn_rate = np.empty(hps.report_every)

    # donate_argnums doesn't seem to make any difference in speed nor memory usage. 
    update_fn_m = jax.pmap(update_fn, axis_name='batch')
    print('Compiled model')
    state = dict(params=params, opt=opt_state, rng=rng_key)
    state_m = flax.jax_utils.replicate(state)
    print('Replicated params across devices')

    step = initial_step 
    dit = dataset.as_numpy_iterator()
    def reshape_fn(ten):
        return jnp.reshape(ten, shape)

    for item in dit:
        item = jax.tree_map(reshape_fn, item)
        inputs = item['inputs']
        targets = item['targets']
        num_toks = inputs['counts'].sum() + targets['counts'].sum()
        state_m, metrics_m = update_fn_m(state_m, inputs, targets) 
        metrics = flax.jax_utils.unreplicate(metrics_m)

        report_idx = step % hps.report_every
        steps[report_idx] = step
        losses[report_idx] = metrics['loss']
        entropies[report_idx] = metrics['entropy']
        learn_rate[report_idx] = learn_rate_fn(step + 1)

        if hps.with_metrics:
            fn = lambda x, y: jax.lax.dynamic_update_index_in_dim(x, y, report_idx, 0)
            norms = jax.tree_map(fn, norms, metrics['norms'])

        if step > 0 and report_idx % hps.report_every == 0:
            loss = metrics['loss']
            entropy = metrics['entropy']
            print(f'step {step}, {num_toks=}, model_entropy={entropy[1]:3.2f} loss={loss:3.2f}')

        if logger and step > 0 and report_idx == hps.report_every - 1:
            log_steps(logger, steps, learn_rate, losses, entropies) 
            if hps.with_metrics:
                log_metrics(logger, steps, norms)

        if (step % hps.ckpt_every == 0 and step > 0 and step != hps.resume_ckpt):
            state = flax.jax_utils.unreplicate(state_m)
            params = state['params']
            state_save_args = jax.tree_map(lambda _: SaveArgs(aggregate=True), params)
            mngr.save(step, params, save_kwargs={'save_args': state_save_args})
            print(f'Saved checkpoint {step=}')
        step += 1

def main(hps_keys: str = 'arch,reg,train,data,logging', **hps_overrides):
    """
    :param resume_ckpt:
        Full path to a checkpoint file.  Use `None` (without quotes) for absent.
        A second line of documentation
        A third line
    :param hps_overrides: Can be any of the following:
           data_dir
              path to dataset prepared using python -m aiayn.data script
    :param streamvis_run_name: name for scoping the run for visualization
    :param pubsub_project: the GCP project with Cloud Pub/Sub API enabled
    :param pubsub_topic: the GCP topic associated with pubsub_project
    :param streamvis_log_file: path to streamvis log file (optional) 

    :param batch_dim0 : SGD batch size dimension 0, the number of sentences in batch
    :param accum_steps: accumulate the gradient over accum_steps steps.
           saves memory for large batch_dim0
           batch_dim0 % accum_steps must be 0
    :param ckpt_dir: directory to save checkpoints
    :param report_every:
           every number of steps to issue progress message to stdout
    :param ckpt_every:
           create a checkpoint every `ckpt_every` steps
    :param with_metrics:
           if True, compute and log additional metrics besides loss
    """

    """
    param_patterns = dict(
        decoder_masked_attention = r'decoder.body.(\d+).mask_att..*',
        decoder_attention2 = r'decoder.body.(\d+).att2..*',
        decoder_feed_forward = r'decoder.body.(\d+).ff..*',
        enc_attention_wq = r'encoder.body.(\d+).att.wq',
        enc_attention_wk = r'encoder.body.(\d+).att.wk',
        enc_attention_wv = r'encoder.body.(\d+).att.wv',
        enc_feed_forward = r'encoder.body.(\d+).ff..*'
        )
    """
    hps = hparams.setup_hparams(hps_keys, hps_overrides)
    print('Now running with parameters:')
    print(hps)

    # This needs to be placed before 
    rng_key = jax.random.PRNGKey(42)

    if hps.streamvis_run_name is not None:
        import streamvis
        logger = streamvis.logger.DataLogger(hps.streamvis_run_name)
        if hps.pubsub_project is not None:
            logger.init_pubsub(hps.pubsub_project, hps.pubsub_topic)
            print(f'Init logger with {hps.pubsub_project} and {hps.pubsub_topic}')
        if hps.streamvis_log_file is not None:
            logger.init_write_log(hps.streamvis_log_file)
            print(f'Init logger write log {hps.streamvis_log_file}')
    else:
        logger = None

    print(f'Prepared dataset from {hps.dataset_glob}')

    # move the save/restore logic here
    update_fn, dataset, params, opt_state, initial_step, mngr, lr_fn = setup_train(hps, rng_key)
    train_loop(hps, update_fn, lr_fn, dataset, params, opt_state, mngr, initial_step,
            rng_key, logger)

if __name__ == '__main__':
    fire.Fire(main)

