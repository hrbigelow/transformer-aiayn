import tensorflow as tf
from tokenizers import Tokenizer
import functools
import psutil
import jax
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
import optax
import haiku as hk
import os
import signal
import queue
import time
import sys
import fire
import numpy as np
from aiayn import model, data, hparams, report, utils, funcs, pack, jutils

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
        # jax.debug.print('step: {}, learn_rate: {}', step, new_lr)
        return new_lr
    return lr_fn

def make_loss_fn(model, objective):
    def loss_fn(params, data, rng):
        inputs = data['inputs']
        targets = data['targets']
        dec_input = targets['seqs']
        dec_output, enc_attn_ent, dec_attn_ent = model.apply(params, rng, inputs, targets)
        metrics = objective.metrics(dec_input, dec_output)

        # TODO: properly normalize each entropy measure by batch size
        metrics.update(enc_attn_entropy=enc_attn_ent, dec_attn_entropy=dec_attn_ent)

        # The loss must be returned in order to use this function to generate
        # gradients.  However, the gradients should be re-normalized, since batch
        # size varies
        loss = objective.loss(metrics)
        return loss, metrics 
    return loss_fn

def accumulate_gradient(loss_fn, params, data, chunk_size, rng):
    """
    Applies loss_fn (and gradient) to `chunk_size` chunks of leading dim of data
    and returns the sum
    """
    # returns (loss, metrics), grad
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    grad_fn = functools.partial(grad_fn, params)

    def reshape_fn(x):
        return jnp.reshape(x, (x.shape[0] // chunk_size, chunk_size, *x.shape[1:]))
    data = jax.tree_map(reshape_fn, data)
    accu, rng = jutils.map_sum(grad_fn, data, rng)
    (_, metrics), grad = accu
    return metrics, grad

def make_validate_fn(model, objective, shard_size):
    loss_fn = make_loss_fn(model, objective)

    def reshape_fn(x):
        return jnp.reshape(x, (x.shape[0] // shard_size, shard_size, *x.shape[1:]))

    def val_fn(state, data, rng_key):
        # compute metrics from state and data
        nonlocal loss_fn
        data = jax.tree_map(reshape_fn, data)
        loss_fn = functools.partial(loss_fn, state['params'])
        (_, metrics), rng_key = jutils.map_sum(loss_fn, data, rng_key)
        metrics = jax.lax.psum(metrics, axis_name='batch')
        metrics = objective.reduce_metrics(metrics)
        return metrics
    return val_fn

def make_update_fn(model, objective, repl_batch_size, accum_steps, with_metrics, tx,
        crash_dir):
    step_size = repl_batch_size // accum_steps

    loss_fn = make_loss_fn(model, objective)

    def update_fn(state, data):
        """
        The function to pass into jax.pmap
        Returns updated state, rng
        """
        """
        Bind the rng key to the device id (which is unique across hosts)
        Note: This is only used for multi-host training (i.e. multiple computers
        each with multiple accelerators).
        """
        params, opt_state, rng_key = state['params'], state['opt_state'], state['rng'] 
        new_rng_key, = jax.random.split(rng_key, 1)
        dropout_rng = jax.random.fold_in(rng_key, jax.lax.axis_index('batch'))

        metrics, grad = accumulate_gradient(loss_fn, params, data, step_size, dropout_rng)
        # metrics, grad = accumulate_gradient(lg_fn, step_size, accum_steps, params,
                # inputs, targets, dropout_rng)

        metrics, grad = jax.lax.psum((metrics, grad), axis_name='batch')
        grad = jax.tree_map(lambda x: x / metrics['sum_active'], grad)
        metrics = objective.reduce_metrics(metrics)
        
        nan_grads = jax.tree_map(lambda x: jnp.any(jnp.isnan(x)), grad)
        got_nan = jnp.any(jnp.stack(jax.tree_util.tree_leaves(nan_grads)))

        # tx.update takes ~6 seconds / 10 steps
        def apply_updates_fn(grad, opt_state, params):
            updates, opt_state = tx.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            return grad, opt_state, params

        def no_op_fn(grad, opt_state, params):
            return grad, opt_state, params

        grad, opt_state, params = jax.lax.cond(got_nan, no_op_fn, apply_updates_fn,
                grad, opt_state, params)

        def tensor_norm(*xs):
            names = ('grad', 'param', 'update')
            return {n:jnp.mean(jax.lax.pow(x, 2.0)) for n,x in zip(names, xs)}

        if with_metrics:
            norms = jax.tree_map(tensor_norm, grad, params, updates)
            state = dict(params=params, opt_state=opt_state, rng=new_rng_key,
                    got_nan=got_nan)
            metrics.update(norms=norms)
            return state, metrics
        else:
            state = dict(params=params, opt_state=opt_state, rng=new_rng_key,
                    got_nan=got_nan)
            return state, metrics
            
    return update_fn

def make_line_plot(logger, label, steps, vals):
    """
    steps: [num_steps]
    vals: [num_steps] or [num_steps, num_lines] 
    Create a tandem_lines plot
    """
    if len(vals.shape) > 1:
        vals = jnp.transpose(vals, (1, 0))
    logger.write(label, x=steps, y=vals)

def log_steps(logger, steps, learn_rate, metrics):
    make_line_plot(logger, 'learn_rate', steps, learn_rate)
    for key, metric in metrics.items():
        make_line_plot(logger, key, steps, metric)

def log_metrics(logger, steps, norms): 

    def make_plot_data(step_data, steps):
        num_values = step_data.shape[0]
        val_steps = jnp.repeat(steps[None,:], num_values, axis=0)
        plot = jnp.stack((val_steps, step_data), axis=2)
        return plot

    def svlog(prefix, label, key, values):
        vals = report.get_layer_values(pfx, values, key)
        plot = make_plot_data(logger, label, steps, vals)
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

    for cat, pfx in cats.items():
        for label, val in norms.items():
            for key in param_map[cat]:
                plot_name = f'{cat}_{label}_{key}'
                svlog(pfx, plot_name, key, val)

def shard_batch(batch):
    nr = jax.local_device_count()
    fn = lambda x: jnp.reshape(x, (nr, x.shape[0] // nr, *x.shape[1:]))
    return jax.tree_map(fn, batch)


def unshard_batch(batch):
    fn = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))
    return jax.tree_map(fn, batch)

def setup_train(hps, rng_key):
    num_replicas = jax.local_device_count()
    if hps.batch_dim0 % num_replicas != 0:
        raise RuntimeError(f'{hps.batch_dim0=} not divisible by {num_replicas=}')
    repl_batch_size = hps.batch_dim0 // num_replicas
    if repl_batch_size % hps.accum_steps != 0:
        raise RuntimeError(f'{repl_batch_size=} not divisible by {hps.accum_steps=}')

    options = ocp.CheckpointManagerOptions(save_interval_steps=hps.ckpt_every, max_to_keep=20)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    mngr = ocp.CheckpointManager(hps.ckpt_dir, checkpointer, options)

    lr_fn = make_learning_rate_fn(hps.warmup_steps, hps.M)
    tx = optax.adam(learning_rate=lr_fn, b1=hps.adam_beta1, b2=hps.adam_beta2,
            eps=hps.adam_eps)

    tokenizer = Tokenizer.from_str(tf.io.gfile.GFile(hps.tokenizer_file).read())
    n_vocab = tokenizer.get_vocab_size() + 2 # add BOS and EOS
    bos_id = tokenizer.token_to_id('[BOS]')
    eos_id = tokenizer.token_to_id('[EOS]')
    pad_id = tokenizer.token_to_id('[PAD]')
    print(f'{pad_id=}')

    mod = model.make_model(hps, bos_id, eos_id, n_vocab, do_batch=True, do_train=True)
    val_mod = model.make_model(hps, bos_id, eos_id, n_vocab, do_batch=True, do_train=False)
    objective = model.Objective(bos_id, n_vocab, hps.label_smooth_eps, hps.attn_loss_weight)

    update_fn = make_update_fn(mod, objective, repl_batch_size, hps.accum_steps,
            hps.with_metrics, tx, hps.ckpt_dir)

    initial_step = int(hps.resume_ckpt or 0)

    num_tries = 10
    feature_lengths = { 'inputs': hps.max_source_len, 'targets': hps.max_target_len }
    train_ds = data.load_tfrecord_dataset(hps.dataset_glob, hps.swap_source_target)
    train_ds = data.add_special_tokens(train_ds, bos_id, eos_id) 
    train_ds = train_ds.repeat().shuffle(hps.shuffle_size, rng_key[0], True)
    train_ds = pack.pack_dataset(train_ds, feature_lengths, 1000, num_tries, pad_id) 
    train_ds = train_ds.batch(hps.batch_dim0)

    # Need to load deterministically so 
    val_ds = data.load_tfrecord_dataset(hps.val_dataset_glob, hps.swap_source_target)
    val_ds = data.add_special_tokens(val_ds, bos_id, eos_id) 

    # This ensures the data will maintain the same order during both packings
    val_ds = val_ds.take(-1).cache()

    pack_ds = pack.pack_dataset(val_ds, feature_lengths, 100, num_tries, pad_id)
    total_packed = sum(1 for _ in pack_ds.as_numpy_iterator())
    # remainder = - ((total_packed + 1000) % - (hps.val_loop_elem * num_replicas))
    remainder = - (total_packed % - (hps.val_loop_elem * num_replicas))
    total_items = total_packed + remainder

    pack_ds = pack.pack_dataset(val_ds, feature_lengths, 100, num_tries, pad_id)
    fill_ds = pack.filler_dataset(feature_lengths, remainder, num_tries, pad_id)

    pack_ds = pack_ds.concatenate(fill_ds)
    val_data = next(pack_ds.batch(total_items).as_numpy_iterator())
    print(val_data['inputs']['seqs'].shape)
    print(f'{total_items=}')

    # Initialize state de-novo
    item = next(train_ds.as_numpy_iterator())
    init_item = jax.tree_map(lambda ten: ten[:1], item)
    params = mod.init(rng_key, init_item['inputs'], init_item['targets'])
    opt_state = tx.init(params)
    state = dict(params=params, opt_state=opt_state)
    initial_step = 0
    
    if hps.resume_ckpt is not None:
        if hps.ckpt_has_last_batch:
            jnp_item = jax.tree_map(jnp.array, item)
            state.update(last_batch=jnp_item)
        shardings = jax.tree_map(lambda x: x.sharding, state)
        restore_args = utils.construct_restore_args(state, shardings)
        state = mngr.restore(hps.resume_ckpt, items=state, restore_kwargs=
                {'restore_args': restore_args})
        initial_step = hps.resume_ckpt

    return mod, val_mod, objective, update_fn, val_data, train_ds, state, initial_step, mngr, lr_fn

def train_loop(hps, mod, val_mod, objective, update_fn, val_data, learn_rate_fn, train_ds, state, mngr,
        initial_step, rng_key, logger):
    num_replicas = jax.local_device_count()
    batch_repl_size = hps.batch_dim0 // num_replicas
    shape = [num_replicas, batch_repl_size, -1]

    steps = np.empty(hps.report_every)
    losses = np.empty(hps.report_every)
    label_entropy = np.empty(hps.report_every)
    cross_entropy = np.empty(hps.report_every)
    enc_attn_entropy = np.empty((hps.report_every, hps.num_layers)) # each layer
    dec_attn_entropy = np.empty((hps.report_every, hps.num_layers)) # each layer
    attn_loss = np.empty(hps.report_every)
    learn_rate = np.empty(hps.report_every)
    report_metrics = dict(
            kldiv=losses,
            label_entropy=label_entropy,
            cross_entropy=cross_entropy,
            # enc_attn_entropy=enc_attn_entropy,
            #dec_attn_entropy=dec_attn_entropy,
            #attn_loss=attn_loss
            )

    if hps.with_metrics: 
        names = ('grad', 'param', 'update')
        fn = lambda x: { l: np.empty(hps.report_every) for l in names } 
        norms = jax.tree_map(fn, state['params'])

    # donate_argnums doesn't seem to make any difference in speed nor memory usage. 
    update_fn_m = jax.pmap(update_fn, axis_name='batch')

    val_fn = make_validate_fn(mod, objective, hps.val_loop_elem)
    val_fn_m = jax.pmap(val_fn, axis_name='batch')

    val_nd_fn = make_validate_fn(val_mod, objective, hps.val_loop_elem)
    val_nd_fn_m = jax.pmap(val_nd_fn, axis_name='batch')
    val_data = shard_batch(val_data)

    print('Compiled model')
    state.update(rng=rng_key)
    state_m = flax.jax_utils.replicate(state)
    print('Replicated state across devices')

    step = initial_step 
    last_batch = state.get('last_batch', None)
    if last_batch is not None:
        dit = [last_batch]
    else:
        dit = train_ds.as_numpy_iterator()

    for item in dit:
        item = shard_batch(item)
        inputs = item['inputs']
        targets = item['targets']
        num_toks = inputs['counts'].sum() + targets['counts'].sum()
        state_m, metrics_m = update_fn_m(state_m, item) 
        metrics = flax.jax_utils.unreplicate(metrics_m)
        got_nan = flax.jax_utils.unreplicate(state_m['got_nan'])

        if got_nan:
            item = unshard_batch(item)
            state.update(last_batch=item)
            save_args = jax.tree_map(lambda _: ocp.SaveArgs(aggregate=True), state)
            mngr.save(step, state, save_kwargs={'save_args': save_args}, force=True)
            # mngr.wait_until_finished()
            raise RuntimeError(f'Got NaN gradients.  Dumping pre-update state at step {step}')

        report_idx = step % hps.report_every
        steps[report_idx] = step
        for key in report_metrics:
            report_metrics[key][report_idx] = metrics[key]
        learn_rate[report_idx] = learn_rate_fn(step + 1)

        if hps.with_metrics:
            fn = lambda x, y: jax.lax.dynamic_update_index_in_dim(x, y, report_idx, 0)
            norms = jax.tree_map(fn, norms, metrics['norms'])

        if step > 0 and report_idx % hps.report_every == 0:
            loss = report_metrics['kldiv'][report_idx]
            # attn_loss = report_metrics['attn_loss'][report_idx]
            ce = report_metrics['cross_entropy'][report_idx]
            print(f'step {step}, {num_toks=}, cross_ent={ce:3.2f} loss={loss:3.2f}')
                     #f' attn_loss={attn_loss:2.4f}')

        if logger and step > 0 and report_idx == hps.report_every - 1:
            log_steps(logger, steps, learn_rate, report_metrics) 
            # jax.debug.print('report:\n{}', report_metrics)
            if hps.with_metrics:
                log_metrics(logger, steps, norms)

        if step > 0 and step % hps.eval_every == 0:
            rng_key_s = jax.random.split(rng_key, num_replicas)
            metrics_m = val_fn_m(state_m, val_data, rng_key_s)
            metrics = flax.jax_utils.unreplicate(metrics_m)
            nd_metrics_m = val_nd_fn_m(state_m, val_data, rng_key_s)
            nd_metrics = flax.jax_utils.unreplicate(nd_metrics_m)
            ce = metrics['cross_entropy']
            ce_nd = nd_metrics['cross_entropy']
            sum_active = metrics['sum_active']
            print(f'step {step}, sum_active={sum_active:d}, ce_val={ce:3.2f}, ce_val_nd={ce_nd:3.2f}')
            if logger:
                logger.write('cross_entropy_val', x=step, y=metrics['cross_entropy']) 
                logger.write('cross_entropy_val_nd', x=step, y=nd_metrics['cross_entropy'])
            

        if (step % hps.ckpt_every == 0 and step != hps.resume_ckpt):
            state = flax.jax_utils.unreplicate(state_m)
            state_save_args = jax.tree_map(lambda _: ocp.SaveArgs(aggregate=True), state)
            mngr.save(step, state, save_kwargs={'save_args': state_save_args})
            # mngr.save(step, state)
            print(f'Saved checkpoint {step=}')
        step += 1

def main(hps_keys: str = 'arch,reg,train,data,logging', **hps_overrides):
    jnp.set_printoptions(precision=2, threshold=100000, edgeitems=100, linewidth=180)
    # import socket
    # host_addr = socket.gethostbyname(socket.gethostname())
    # jax.distributed.initialize(f'{host_addr}:1234', num_processes=1, process_id=0)

    hps = hparams.setup_hparams(hps_keys, hps_overrides)
    print('Now running with parameters:')
    print(hps)

    # This needs to be placed before 
    rng_key = jax.random.PRNGKey(hps.random_seed)

    if hps.streamvis_run_name is not None:
        from streamvis.logger import DataLogger
        logger = DataLogger(hps.streamvis_run_name)
        logger.init(hps.streamvis_path, hps.streamvis_buffer_items)
    else:
        logger = None

    print(f'Prepared dataset from {hps.dataset_glob}')

    # move the save/restore logic here
    mod, val_mod, objective, update_fn, val_data, train_ds, state, initial_step, mngr, lr_fn = setup_train(hps, rng_key)
    train_loop(hps, mod, val_mod, objective, update_fn, val_data, lr_fn, train_ds, state, mngr,
            initial_step, rng_key, logger)

if __name__ == '__main__':
    fire.Fire(main)

