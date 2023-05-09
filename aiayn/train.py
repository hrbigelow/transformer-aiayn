import fire
import numpy as np
from . import model
from . import data
from . import state
import torch
from torch.optim import Adam
from streamvis import DataLogger 

def main(batch_size, sub_batch_size, data_path, ckpt_templ, max_sentence_length,
        report_every=10, ckpt_every=1000, resume_ckpt=None, compile_model=False,
        pubsub_project: str = None, streamvis_log_file: str = None):
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
    if pubsub_project is None and streamvis_log_file is None:
        raise RuntimeError(
            f'At least one of `pubsub_project` or `streamvis_log_file` must be provided')

    if batch_size % sub_batch_size != 0:
        raise RuntimeError(f'batch_size = {batch_size} must be disivible by '
                f'sub_batch_size = {sub_batch_size}')

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

    logger.clear()

    # (top, left, height, width)
    grid_map = dict(
            loss = (0,0,1,1),
            decoder_masked_attention = (0,1,1,1),
            decoder_attention2 = (1,0,1,1),
            decoder_feed_forward = (1,1,1,1),
            enc_attention_wq = (2,0,1,1),
            enc_attention_wk = (2,1,1,1),
            enc_attention_wv = (2,2,1,1),
            enc_feed_forward = (2,3,1,1))
    logger.set_layout(grid_map)

    max_height, max_width = 900, 1800
    grad_kwargs = dict(height=300, width=1800//5)
    loss_kwargs = dict(height=300, width=1800)
    cell_kwargs = dict(height=max_height // 3, width=max_width // 2)

    run = state.Run(
            model=model.StateModel(),
            data=data.Data(),
            opt=model.StateOptim(),
            sched=model.CustomScheduler()) 
    run.add_deps('sched', 'opt')
    run.add_deps('opt', 'model')

    ds = data.get_dataset(data_path)
    tokenizer = data.get_tokenizer()
    print(f'Prepared dataset')

    if resume_ckpt is None:
        hps = model.HyperParams()
        hps.T = len(tokenizer)
        hps.batch_size = batch_size
        hps.sub_batch_size = sub_batch_size
        hps.dataset_size = len(ds)
        hps.data_path = data_path
        hps.pad_token_id = tokenizer.pad_token_id
        run.init(hps)
    else:
        path = ckpt_templ.format(resume_ckpt)
        print(f'Resuming from {path}')
        run.load(path)

    if torch.cuda.get_device_capability() >= (7,0):
        if compile_model:
            print('Compiling model')
            run.model = torch.compile(run.model)

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
        if longest_sentence > max_sentence_length:
            print(f'step {step}: skipping en = {en_range}, de = {de_range}')
            continue

        lr = run.sched.current_lr()
        if step % report_every == 0:
            print(f'epoch = {epoch}, step = {step}, '
                    f'en = {en_range}, de = {de_range}, '
                    f'lr = {lr:7.6f}', end='', flush=True) 
        
        enc_input = enc_input.reshape(*batch_shape, *enc_input.shape[1:])
        dec_input = dec_input.reshape(*batch_shape, *dec_input.shape[1:])
        run.model.zero_grad()

        sub_loss = torch.zeros(batch_shape[0])

        for sub_batch in range(batch_shape[0]):
            sub_enc_input = enc_input[sub_batch]
            sub_dec_input = dec_input[sub_batch]
            sub_dec_output = run.model(sub_enc_input, sub_dec_input)
            xent = run.model.loss(sub_dec_input, sub_dec_output)
            xent.backward()
            with torch.no_grad():
                sub_loss[sub_batch] = xent.item()

        run.opt.step()
        run.sched.update(step)

        loss = sub_loss.mean().item()
        loss_metrics = [loss, en_lengths.sum().item() / loss, de_lengths.sum().item() / loss]
        logger.tandem_lines('loss', step, loss_metrics, 'Viridis256', fig_kwargs=cell_kwargs)

        for plot, pattern in param_patterns.items():
            norms = run.model.grad_norms(pattern) 
            logger.tandem_lines(plot, step, norms, 'Viridis256', fig_kwargs=cell_kwargs)

        if step % report_every == 0:
            print(f', loss = {loss:5.4f}', flush=True)

        if step % ckpt_every == 0 and step > 0 and step != resume_ckpt:
            path = ckpt_templ.format(step)
            print(f'Saving {path}', flush=True)
            run.save(path)

if __name__ == '__main__':
    fire.Fire(main)


