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
from aiayn import model, data, hparams, report, utils, funcs, pack

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
        nan_grads = jax.tree_map(lambda x: jnp.any(jnp.isnan(x)), gi)
        got_nan = jnp.any(jnp.stack(jax.tree_util.tree_leaves(nan_grads)))
        # jax.debug.print('got nan: {}', got_nan)
        # flattened, _ = jax.tree_util.tree_flatten_with_path(nan_grads)
        # flattened = [(jax.tree_util.keystr(k), v) for k, v in flattened]
        # jax.debug.print('nan_grads:\n{}', nan_grads)
        # for path, val in flattened:
            # jax.debug.print('{path}: {}', val, path=jax.tree_util.keystr(path))
        l, e, g = tup
        return (l + li, map_add(e, ei), map_add(g, gi))

    # 1 accum:  6 sec / 10 steps
    # 5 accum: 9 sec / 10 steps
    # 9 accum: 14 sec / 10 steps 
    l, e, g = jax.lax.fori_loop(1, accum_steps, acc_loss_and_grad, (l, e, g))
    return jax.tree_map(lambda x: x / accum_steps, (l, e, g))

def make_update_fn(model, objective, repl_batch_size, accum_steps, with_metrics, tx,
        crash_dir):
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
        params, opt_state, rng_key = state['params'], state['opt_state'], state['rng'] 
        _, new_rng_key = jax.random.split(rng_key)
        dropout_rng = jax.random.fold_in(rng_key, jax.lax.axis_index('batch'))

        def loss_fn(params, inputs, targets):
            dec_input = targets['seqs']
            dec_output, enc_attn_ent, dec_attn_ent = model.apply(params, dropout_rng,
                    inputs, targets)
            # print_range('dec_output', dec_output)
            metrics = objective.metrics(dec_input, dec_output, enc_attn_ent, dec_attn_ent)
            # jax.debug.print('bla-metrics:{}', metrics)
            loss = objective.loss(metrics['kldiv'], metrics['attn_loss'])
            return loss, metrics 

        lg_fn = jax.value_and_grad(loss_fn, has_aux=True)

        # setting accum_steps to 1 results in 
        l, metrics, g = accumulate_gradient(lg_fn, step_size, accum_steps, params,
                inputs, targets)
        # averages l and g across replicas, then broadcasts the average
        loss, metrics, grad = jax.lax.pmean((l, metrics, g), axis_name='batch')

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
            metrics = {'loss': loss, 'norms': norms, **metrics}
            return state, metrics
        else:
            state = dict(params=params, opt_state=opt_state, rng=new_rng_key,
                    got_nan=got_nan)
            metrics = {'loss': loss, **metrics} 
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

def make_tandem_plot(logger, key, steps, values, palette):
    """
    Create a tandem lines plot with name `key`
    steps:  b
    values: bl 
    b lines, each with l points
    line i, point p is (steps[i], values[i][p])
    """
    steps = jnp.tile(steps[:,None], values.shape[1])
    plot = jnp.stack((steps, values), axis=2)
    plot = jnp.transpose(plot, (1,0,2))
    # jax.debug.print('plot:\n{}', plot) # bl2  
    logger.tandem_lines(key, plot, palette=palette)

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

def setup_train(hps, rng_key):
    num_replicas = jax.local_device_count()
    if hps.batch_dim0 % num_replicas != 0:
        raise RuntimeError(f'{hps.batch_dim0=} not divisible by {num_replicas=}')
    repl_batch_size = hps.batch_dim0 // num_replicas
    if repl_batch_size % hps.accum_steps != 0:
        raise RuntimeError(f'{repl_batch_size=} not divisible by {hps.accum_steps=}')

    # Set up orbax to save/restore custom optax types
    # utils.register_handlers()

    options = ocp.CheckpointManagerOptions(save_interval_steps=hps.ckpt_every, max_to_keep=20)
    # checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler(), timeout_secs=100)
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

    mod = model.make_model(hps, bos_id, eos_id, n_vocab, True)
    objective = model.Objective(bos_id, n_vocab, hps.label_smooth_eps, hps.attn_loss_weight)

    update_fn = make_update_fn(mod, objective, repl_batch_size, hps.accum_steps,
            hps.with_metrics, tx, hps.ckpt_dir)

    initial_step = int(hps.resume_ckpt or 0)

    feature_lengths = { 'inputs': hps.max_source_len, 'targets': hps.max_target_len }
    token_ds = data.load_tfrecord_dataset(hps.dataset_glob, hps.swap_source_target)
    token_ds = data.add_special_tokens(token_ds, bos_id, eos_id) 
    token_ds = token_ds.repeat().shuffle(hps.shuffle_size, rng_key[0], True)
    pack_ds = pack.pack_dataset(token_ds, feature_lengths, 1000, 10, pad_id) 
    dataset = pack_ds.rebatch(hps.batch_dim0)

    # Initialize state de-novo
    item = next(dataset.as_numpy_iterator())
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

    return update_fn, dataset, state, initial_step, mngr, lr_fn

def shard_batch(batch):
    nr = jax.local_device_count()
    fn = lambda x: jnp.reshape(x, (nr, x.shape[0] // nr, *x.shape[1:]))
    return jax.tree_map(fn, batch)

def unshard_batch(batch):
    fn = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))
    return jax.tree_map(fn, batch)

def train_loop(hps, update_fn, learn_rate_fn, dataset, state, mngr, initial_step, rng_key,
        logger):
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
            enc_attn_entropy=enc_attn_entropy,
            dec_attn_entropy=dec_attn_entropy,
            attn_loss=attn_loss
            )

    if hps.with_metrics: 
        names = ('grad', 'param', 'update')
        fn = lambda x: { l: np.empty(hps.report_every) for l in names } 
        norms = jax.tree_map(fn, state['params'])

    # donate_argnums doesn't seem to make any difference in speed nor memory usage. 
    update_fn_m = jax.pmap(update_fn, axis_name='batch')
    print('Compiled model')
    state.update(rng=rng_key)
    state_m = flax.jax_utils.replicate(state)
    print('Replicated state across devices')

    step = initial_step 
    last_batch = state.get('last_batch', None)
    if last_batch is not None:
        dit = [last_batch]
    else:
        dit = dataset.as_numpy_iterator()

    for item in dit:
        item = shard_batch(item)
        inputs = item['inputs']
        targets = item['targets']
        num_toks = inputs['counts'].sum() + targets['counts'].sum()
        state_m, metrics_m = update_fn_m(state_m, inputs, targets) 
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
            attn_loss = report_metrics['attn_loss'][report_idx]
            ce = report_metrics['cross_entropy'][report_idx]
            print(f'step {step}, {num_toks=}, cross_ent={ce:3.2f} loss={loss:3.2f}'
                    f' attn_loss={attn_loss:2.4f}')

        if logger and step > 0 and report_idx == hps.report_every - 1:
            log_steps(logger, steps, learn_rate, report_metrics) 
            # jax.debug.print('report:\n{}', report_metrics)
            if hps.with_metrics:
                log_metrics(logger, steps, norms)

        if (step % hps.ckpt_every == 0 and step != hps.resume_ckpt):
            state = flax.jax_utils.unreplicate(state_m)
            state_save_args = jax.tree_map(lambda _: ocp.SaveArgs(aggregate=True), state)
            mngr.save(step, state, save_kwargs={'save_args': state_save_args})
            # mngr.save(step, state)
            print(f'Saved checkpoint {step=}')
        step += 1

def main(hps_keys: str = 'arch,reg,train,data,logging', **hps_overrides):
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
    jnp.set_printoptions(precision=2, threshold=100000, edgeitems=100, linewidth=180)
    # import socket
    # host_addr = socket.gethostbyname(socket.gethostname())
    # jax.distributed.initialize(f'{host_addr}:1234', num_processes=1, process_id=0)

    hps = hparams.setup_hparams(hps_keys, hps_overrides)
    print('Now running with parameters:')
    print(hps)

    # This needs to be placed before 
    rng_key = jax.random.PRNGKey(42)

    if hps.streamvis_run_name is not None:
        from streamvis.logger import DataLogger
        logger = DataLogger(hps.streamvis_run_name)
        logger.init(hps.streamvis_path, hps.streamvis_buffer_items)
    else:
        logger = None

    print(f'Prepared dataset from {hps.dataset_glob}')

    # move the save/restore logic here
    update_fn, dataset, state, initial_step, mngr, lr_fn = setup_train(hps, rng_key)
    train_loop(hps, update_fn, lr_fn, dataset, state, mngr, initial_step, rng_key,
            logger)

if __name__ == '__main__':
    fire.Fire(main)

