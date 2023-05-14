import signal
import sys
import fire
import numpy as np
from aiayn import model, data, state
import torch
from torch.optim import Adam
from streamvis import DataLogger 
from aiayn.hparams import setup_hparams, Hyperparams

import pdb

def main(hps_keys='arch,reg,train,data', 
        data_path: str = None, 
        pubsub_project: str = None, 
        streamvis_log_file: str = None, 
        **kwargs):
    """
    :param batch_size: SGD batch size
    :param sub_batch_size: size used for a single forward pass to accumulate
                           gradients.  must be factor of batch_size
    :param data_path:
           path to dataset prepared using python -m aiayn.data script
    :param ckpt_templ: checkpoint file path containing literal {} to be substituted with 
                       step value
    :param max_sentence_length: skip data batches containing token sequences longer
                                than this, to avoid OOM errors
    :param report_every:
           every number of steps to issue progress message to stdout
    :param ckpt_every:
           create a checkpoint every `ckpt_every` steps
    :param resume_ckpt:
           if present, resume from this checkpoint file
    :param compile_model: obsolete (varying sentence length seems to prevent
                                    effective compilation)
    :param pubsub_project: the GCP project with Cloud Pub/Sub API enabled
    :param streamvis_log_file: path to streamvis log file (optional) 

    """
    if 'ckpt_file' in kwargs:
        hps = Hyperparams(kwargs)
        if 'random_seed' not in hps:
            hps.random_seed = 2507
    else:
        kwargs['data_path'] = data_path
        hps = setup_hparams(hps_keys, kwargs)

    if pubsub_project is None and streamvis_log_file is None:
        raise RuntimeError(
            f'At least one of `pubsub_project` or `streamvis_log_file` must be provided')

    if hps.batch_size % hps.sub_batch_size != 0:
        raise RuntimeError(f'{hps.batch_size=} must be disivible by {hps.sub_batch_size=}')

    param_patterns = dict(
        decoder_masked_attention = r'decoder.body.(\d+).mask_att..*',
        decoder_attention2 = r'decoder.body.(\d+).att2..*',
        decoder_feed_forward = r'decoder.body.(\d+).ff..*',
        enc_attention_wq = r'encoder.body.(\d+).att.wq',
        enc_attention_wk = r'encoder.body.(\d+).att.wk',
        enc_attention_wv = r'encoder.body.(\d+).att.wv',
        enc_feed_forward = r'encoder.body.(\d+).ff..*'
        )

    logger = DataLogger('aiayn')
    if pubsub_project is not None:
        logger.init_pubsub(pubsub_project, 'aiayn')

    if streamvis_log_file is not None:
        logger.init_write_log(streamvis_log_file)

    run = state.Run(
            model=model.StateModel(),
            data=data.Data(),
            opt=model.StateOptim(),
            sched=model.CustomScheduler()) 
    run.add_deps('sched', 'opt')
    run.add_deps('opt', 'model')

    ds = data.get_dataset(hps.data_path)
    tokenizer = data.get_tokenizer()
    print(f'Prepared dataset')

    if hps.resume_ckpt:
        path = hps.ckpt_templ.format(hps.resume_ckpt)
        print(f'Resuming from {path}')
        run.load(path)
    else:
        hps.T = len(tokenizer)
        hps.dataset_size = len(ds)
        hps.pad_token_id = tokenizer.pad_token_id
        run.init(hps)

    if torch.cuda.get_device_capability() >= (7,0):
        if hps.compile_model:
            print('Compiling model')
            run.model = torch.compile(run.model)

    print(hps)

    def shutdown_handler(signum, frame):
        logger.shutdown()
        path = hps.ckpt_templ.format(run.sched.step)
        run.save(path)
        print(f'Saved final checkpoint to {path}')
        print(f'Exiting after receiving {signal.Signals(signum).name}')
        sys.exit(0)

    # signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    run.to(torch.device('cuda'))
    run.sched.update(0)

    batch_shape = (hps.batch_size // hps.sub_batch_size, hps.sub_batch_size)

    inds_gen = iter(run.data)
    for epoch, step, inds in inds_gen:
        enc_input, dec_input, en_lengths, de_lengths = data.make_batch(ds, inds, 
                tokenizer.pad_token_id, 'cuda')
        mean_en = en_lengths.to(torch.float32).mean()
        mean_de = de_lengths.to(torch.float32).mean()
        # print(mean_en)
        en_range = (en_lengths.min().item(), en_lengths.max().item())
        de_range = (de_lengths.min().item(), de_lengths.max().item())
        longest_sentence = max(en_range[1], de_range[1])
        if longest_sentence > hps.max_sentence_length:
            print(f'step {step}: skipping en = {en_range}, de = {de_range}')
            continue

        lr = run.sched.current_lr()
        if step % hps.report_every == 0:
            print(f'epoch = {epoch}, step = {step}, '
                    f'en = {en_range}, de = {de_range}, '
                    f'lr = {lr:7.6f}', end='', flush=True) 
        
        enc_input = enc_input.reshape(*batch_shape, *enc_input.shape[1:])
        dec_input = dec_input.reshape(*batch_shape, *dec_input.shape[1:])

        sub_loss = torch.zeros(batch_shape[0])

        enc_layer0 = r'encoder.body.0.att.wq'
        layer0_norms = torch.empty(batch_shape)

        # accumulate gradients over sub-batches
        batch_fraction = hps.sub_batch_size / hps.batch_size
        run.model.zero_grad()
        for sub_batch in range(batch_shape[0]):
            sub_enc_input = enc_input[sub_batch]
            sub_dec_input = dec_input[sub_batch]
            sub_dec_output = run.model(sub_enc_input, sub_dec_input)

            # zero out specific gradients for later inspection
            layer0_grads = run.model.zero_gradients(enc_layer0)

            # scale loss by batch_fraction
            xent = run.model.loss(sub_dec_input, sub_dec_output) * batch_fraction

            # copy gradients
            xent.backward()

            # norms: pat, B (only one pat here)
            norms = run.model.grad_norms(enc_layer0, index_dims=(0,)) 
            layer0_norms[sub_batch] = norms[0]

            # add back copied gradients
            run.model.add_gradients(layer0_grads)

            with torch.no_grad():
                sub_loss[sub_batch] = xent.item()

        # protect to avoid partial update
        old_handler = signal.signal(signal.SIGTERM, None)
        run.opt.step()
        run.sched.update(step)
        signal.signal(signal.SIGTERM, old_handler)

        loss = sub_loss.mean().item()
        loss_metrics = [loss]
        logger.tandem_lines('loss', step, loss_metrics, 'Viridis256')
        logger.tandem_lines('lr', step, [lr], 'Viridis256')
        logger.tandem_lines('epoch', step, [epoch])
        logger.tandem_lines('en_lengths', step, en_lengths, 'Viridis256')
        logger.tandem_lines('de_lengths', step, de_lengths, 'Viridis256')

        layer0_norms = layer0_norms.reshape(hps.batch_size).cpu().numpy()
        logger.tandem_lines('enc_layer0', step, layer0_norms, 'Viridis256')
        """
        for plot, pattern in param_patterns.items():
            norms = run.model.grad_norms(pattern) 
            logger.tandem_lines(plot, step, norms, 'Viridis256')
        """

        if step % hps.report_every == 0:
            print(f', loss = {loss:5.4f}', flush=True)

        if step % hps.ckpt_every == 0 and step > 0 and step != hps.resume_ckpt:
            path = hps.ckpt_templ.format(step)
            print(f'Saving {path}', flush=True)
            run.save(path)

if __name__ == '__main__':
    fire.Fire(main)

