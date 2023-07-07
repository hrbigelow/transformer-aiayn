import psutil
import jax
import jax.numpy as jnp
import flax
import orbax.checkpoint as orbax
import optax
import haiku as hk
import os
import signal
import queue
import time
import sys
import fire
import numpy as np
from aiayn import model, data, hparams, report

def print_tree_summary(tree, summary_fn):
    def join_path(path):
        return '->'.join([el.key for el in path])
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    summary = [(join_path(path), summary_fn(leaf)) for path, leaf in flat]
    for name, val in summary:
        jax.debug.print(name + ': {}', val)

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

def accumulate_gradient(loss_and_grad_fn, params, enc_input, dec_input, accum_steps):
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
    if accum_steps and accum_steps > 1:
        assert enc_input.shape[0] % accum_steps == 0, (
               f'batch size {enc_input.shape[0]} not a multiple of {accum_steps=}')

        step_size = enc_input.shape[0] // accum_steps
        l, g = loss_and_grad_fn(params, enc_input[:step_size], dec_input[:step_size])

        def acc_grad_and_loss(i, l_and_g):
            encs = jax.lax.dynamic_slice(enc_input, (i * step_size, 0),
                    (step_size,) + enc_input.shape[1:])
            decs = jax.lax.dynamic_slice(dec_input, (i * step_size, 0),
                    (step_size,) + dec_input.shape[1:])
            li, gi = loss_and_grad_fn(params, encs, decs)
            l, g = l_and_g
            return (l + li, jax.tree_map(lambda x, y: x + y, g, gi))

        l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
        return jax.tree_map(lambda x: x / accum_steps, (l, g))

    else:
        return loss_and_grad_fn(params, enc_input, dec_input)


def make_update_fn(model, objective, accum_steps, with_metrics, tx):
    def update_fn(params, opt_state, enc_input, dec_input, rng_key):
        """
        The function to pass into jax.pmap
        Returns updated state, rng
        """
        """
        Bind the rng key to the device id (which is unique across hosts)
        Note: This is only used for multi-host training (i.e. multiple computers
        each with multiple accelerators).
        """
        _, new_rng_key = jax.random.split(rng_key)
        dropout_rng = jax.random.fold_in(rng_key, jax.lax.axis_index('batch'))

        def loss_fn(params, enc_input, dec_input):
            dec_output = model.apply(params, dropout_rng, enc_input, dec_input)
            # print_range('dec_output', dec_output)
            loss = objective.apply(None, None, dec_input, dec_output)
            return loss

        def tensor_norm(x):
            return jnp.mean(jax.lax.pow(x, 2.0))

        lg_fn = jax.value_and_grad(loss_fn)
        l, g = accumulate_gradient(lg_fn, params, enc_input, dec_input, accum_steps)
        # averages l and g across replicas, then broadcasts the average
        l = jax.lax.pmean(l, axis_name='batch')
        g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)

        updates, opt_state = tx.update(g, opt_state)
        params = optax.apply_updates(params, updates)

        if with_metrics:
            gnorm = jax.tree_map(tensor_norm, g)
            pnorm = jax.tree_map(tensor_norm, params)
            unorm = jax.tree_map(tensor_norm, updates)
            return params, opt_state, l, new_rng_key, (gnorm, pnorm, unorm)
        else:
            return params, opt_state, l, new_rng_key, None
            
    return update_fn

def log_steps(logger, steps, learn_rate, losses, learn_rates):
    loss_plot = jnp.expand_dims(jnp.stack((steps, losses), axis=1), axis=0)
    logger.tandem_lines('loss', loss_plot)
    lr_plot = jnp.expand_dims(jnp.stack((steps, learn_rate), axis=1), axis=0)
    logger.tandem_lines('learn_rate', lr_plot)

def log_metrics(logger, steps, grad_norms, param_norms, update_norms): 

    def make_plot_data(step_data, steps):
        num_values = step_data.shape[0]
        val_steps = jnp.repeat(jnp.expand_dims(steps, axis=0), num_values, axis=0)
        plot = jnp.stack((val_steps, step_data), axis=2)
        return plot

    def svlog(prefix, label, key, values):
        vals = report.get_layer_values(pfx, values, key)
        plot = make_plot_data(vals, steps)
        logger.tandem_lines(label, plot, palette='Viridis256')
   
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

    vals_map = {
            'grad_norm': grad_norms,
            'param_norm': param_norms,
            'update_norm': update_norms
            }

    for cat, pfx in cats.items():
        for label, val in vals_map.items():
            for key in param_map[cat]:
                plot_name = f'{cat}_{label}_{key}'
                svlog(pfx, plot_name, key, val)

def restore_params(ckpt_path):
    zfile = np.load(ckpt_path)

def train_loop(hps, model, learn_rate_fn, objective, tx, dataset, rng_key, logger):
    num_replicas = jax.local_device_count()
    batch_repl_size = hps.batch_dim0 // num_replicas
    shape = [num_replicas, batch_repl_size, -1]
    enc_input, dec_input, _, _ = next(iter(dataset))
    enc_input = enc_input.reshape(shape)
    dec_input = dec_input.reshape(shape)
    
    options = orbax.CheckpointManagerOptions(save_interval_steps=hps.ckpt_every, 
            max_to_keep=10)
    mngr = orbax.CheckpointManager(
        hps.ckpt_dir, orbax.Checkpointer(orbax.PyTreeCheckpointHandler()), options)

    if hps.resume_ckpt:
        params = mngr.restore(hps.resume_ckpt)
        opt_state = tx.init(params)
        initial_step = hps.resume_ckpt
    else:
        params = model.init(rng_key, enc_input[0], dec_input[0])
        opt_state = tx.init(params)
        initial_step = 0
    print('Initialized model')

    if hps.with_metrics: 
        grad_norms = jax.tree_map(lambda x: np.empty(hps.report_every), params)
        param_norms = jax.tree_map(lambda x: np.empty(hps.report_every), params)
        update_norms = jax.tree_map(lambda x: np.empty(hps.report_every), params)

    steps = np.empty(hps.report_every)
    losses = np.empty(hps.report_every)
    learn_rate = np.empty(hps.report_every)

    update_fn = make_update_fn(model, objective, hps.accum_steps, hps.with_metrics, tx)
    # donate_argnums doesn't seem to make any difference in speed
    # nor memory usage. 
    update_fn_repl = jax.pmap(update_fn, axis_name='batch')
    print('Compiled model')
    params_repl = flax.jax_utils.replicate(params)
    opt_state_repl = flax.jax_utils.replicate(opt_state)
    rng_key_repl = flax.jax_utils.replicate(rng_key)
    print('Replicated params across devices')

    def update_elem(tensor, idx, elem_value):
        # performs tensor[idx] = elem_value but on immutable tensor
        fun = lambda x, y: jax.lax.dynamic_update_index_in_dim(x, y, idx, 0)
        return jax.tree_map(fun, tensor, elem_value)

    step = initial_step 

    for enc_input, dec_input, enc_lengths, dec_lengths in iter(dataset):
        num_toks = enc_lengths.sum() + dec_lengths.sum()
        enc_input = enc_input.reshape(shape)
        dec_input = dec_input.reshape(shape)
        params_repl, opt_state_repl, loss_repl, rng_key_repl, metrics = (
            update_fn_repl(params_repl, opt_state_repl, enc_input, dec_input, rng_key_repl))

        loss = flax.jax_utils.unreplicate(loss_repl)

        report_idx = step % hps.report_every
        steps[report_idx] = step
        losses[report_idx] = loss
        learn_rate[report_idx] = learn_rate_fn(step)

        if metrics:
            gnorm, pnorm, unorm = tuple(flax.jax_utils.unreplicate(m) for m in metrics)
            grad_norms = update_elem(grad_norms, report_idx, gnorm)
            param_norms = update_elem(param_norms, report_idx, pnorm)
            update_norms = update_elem(update_norms, report_idx, unorm)

        if step > 0 and report_idx % hps.report_every == 0:
            print(f'step {step}, {num_toks=}, loss={loss:3.2f}')

        if logger and step > 0 and report_idx == hps.report_every - 1:
            log_steps(logger, steps, learn_rate, losses, None) 
            if hps.with_metrics:
                log_metrics(logger, steps, grad_norms, param_norms, update_norms)

        if (step % hps.ckpt_every == 0 and step > 0 and step != hps.resume_ckpt):
            params = flax.jax_utils.unreplicate(params_repl) 
            state_save_args = jax.tree_map(lambda _: orbax.SaveArgs(aggregate=True), params)
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
           data_path
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
    :param max_sentence_length: skip data batches containing token sequences longer
                                than this, to avoid OOM errors
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

    dataset, ds_info = data.base_dataset(hps.data_path, 'train', 2)
    dataset = data.pipe_dataset(dataset, ds_info, hps.max_sentence_length, hps.batch_dim0)

    lr_fn = make_learning_rate_fn(hps.warmup_steps, hps.M)
    tx = optax.chain(
            optax.adam(learning_rate=lr_fn, b1=hps.adam_beta1, b2=hps.adam_beta2,
                eps=hps.adam_eps)
            )

    token_info = data.load_token_info(hps.data_path) 
    mod = model.make_model(hps, True, token_info)
    objective = model.make_objective(token_info)

    train_loop(hps, mod, lr_fn, objective, tx, dataset, rng_key, logger)

if __name__ == '__main__':
    fire.Fire(main)

