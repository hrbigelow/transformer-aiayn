import psutil
import jax
import jax.numpy as jnp
import flax
import optax
import haiku as hk
import os
import signal
import queue
import time
import sys
import fire
import numpy as np
from aiayn import model, data, hparams

def report_fn(logger, epoch, steps, loss, learn_rates):
    if logger is None:
        return
    if xm.is_master_ordinal():
        # 1, R, 2
        steps = steps.cpu()
        loss = loss.cpu()
        learn_rates = learn_rates.cpu()
        loss_plot = t.stack((steps, loss), dim=1).unsqueeze(0)
        logger.tandem_lines('loss', loss_plot)

        lr_plot = t.stack((steps, learn_rates), dim=1).unsqueeze(0)
        logger.tandem_lines('lr', lr_plot)
        print(f'{time.time():.0f}: {epoch=}, {steps=}, {loss=}')

def element_mean_fn(tensors):
    return t.stack(tensors).to(t.float32).mean(dim=0)

def make_learning_rate_fn(warmup_steps, M):
    # from section 5.3, page 7, equation 3
    def lr_fn(step):
        factor = min(step ** -0.5, step * warmup_steps ** -1.5)
        new_lr = M ** -0.5 * factor
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


def make_update_fn(model, objective, accum_steps, tx):
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
            loss = objective.apply(None, None, dec_input, dec_output)
            return loss

        lg_fn = jax.value_and_grad(loss_fn)
        l, g = accumulate_gradient(lg_fn, params, enc_input, dec_input, accum_steps)
        # averages l and g across replicas, then broadcasts the average
        l = jax.lax.pmean(l, axis_name='batch')
        g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
        updates, opt_state = tx.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l, new_rng_key
    return update_fn

def report(logger, epoch, steps, losses):
    loss_plot = jnp.stack((steps, loss), dim=1).unsqueeze(0)
    logger.tandem_lines('loss', loss_plot)
    lr_plot = jnp.stack((steps, learn_rates), dim=1).unsqueeze(0)
    logger.tandem_lines('lr', lr_plot)

def restore_params(ckpt_path):
    zfile = np.load(ckpt_path)

def train_loop(hps, model, objective, tx, dataset, rng_key):
    num_replicas = jax.local_device_count()
    batch_repl_size = hps.batch_size // num_replicas
    shape = [num_replicas, batch_repl_size, -1]
    enc_input, dec_input = next(iter(dataset))
    enc_input = enc_input.reshape(shape)
    dec_input = dec_input.reshape(shape)
    
    if hps.resume_ckpt:
        pass
    else:
        params = model.init(rng_key, enc_input[0], dec_input[0])
        opt_state = tx.init(params)
        initial_step = 0
    print('Initialized model')

    update_fn = make_update_fn(model, objective, hps.accum_steps, tx)
    # TODO: how to use donate_argnums? 
    update_fn_repl = jax.pmap(update_fn, axis_name='batch')
    print('Compiled model')
    params_repl = flax.jax_utils.replicate(params)
    opt_state_repl = flax.jax_utils.replicate(opt_state)
    rng_key_repl = flax.jax_utils.replicate(rng_key)
    print('Replicated params across devices')

    step = initial_step 
    for enc_input, dec_input in iter(dataset):
        enc_input = enc_input.reshape(shape)
        dec_input = dec_input.reshape(shape)
        params_repl, opt_state_repl, loss, rng_key_repl = (
                update_fn_repl(params_repl, opt_state_repl, enc_input, dec_input,
                    rng_key_repl))
        print(f'step {step}, loss={loss:%3.2f}')

        if step > 0 and report ==  hps.report_every - 1:
            report(logger, epoch, steps, losses) 

        if (step % hps.ckpt_every == 0 and step > 0 and step != hps.resume_ckpt):
            path = hps.ckpt_templ.format(step)
            params = flax.jax_utils.unreplicate(params_repl) 
            # save(path)

def main(resume_ckpt, hps_keys: str = 'arch,reg,train,data,logging', **hps_overrides):
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

    :param batch_size: SGD batch size
    :param update_every: number of loader steps to accumulate gradients for before
                         taking an optimizer step
    :param ckpt_templ: checkpoint file path containing literal {} to be substituted with 
                       step value
    :param max_sentence_length: skip data batches containing token sequences longer
                                than this, to avoid OOM errors
    :param report_every:
           every number of steps to issue progress message to stdout
    :param ckpt_every:
           create a checkpoint every `ckpt_every` steps
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

    rng_key = jax.random.PRNGKey(42)

    # if hps.pubsub_project is None and hps.streamvis_log_file is None:
        # raise RuntimeError(
            # f'At least one of `pubsub_project` or `streamvis_log_file` must be provided')
    dataset, ds_info = data.base_dataset(hps.data_path, 'train', 2)
    dataset = data.pipe_dataset(dataset, ds_info, hps.max_sentence_length, hps.batch_size)

    # lr_fn = make_learning_rate_fn(hps.warmup_steps, hps.M)
    lr_fn = 0.001 
    tx = optax.chain(
            optax.adam(learning_rate=lr_fn, b1=hps.adam_beta1, b2=hps.adam_beta2,
                eps=hps.adam_eps)
            )

    token_info = data.load_token_info(hps.data_path) 
    mod = model.make_model(hps, True, token_info)
    objective = model.make_objective(token_info)

    train_loop(hps, mod, objective, tx, dataset, rng_key)

if __name__ == '__main__':
    fire.Fire(main)

