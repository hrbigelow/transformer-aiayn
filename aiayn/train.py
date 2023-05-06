import fire
import numpy as np
from . import model
from . import data
from . import state
import torch
from torch.optim import Adam
from streamvis import Client

def main(pubsub_project_id, batch_size, sub_batch_size, data_path, ckpt_templ,
        report_every=10, ckpt_every=1000, resume_ckpt=None, compile_model=False):
    """
    pubsub_project_id: the GCP project with Cloud Pub/Sub API enabled
    batch_size: SGD batch size
    sub_batch_size: size used for a single forward pass to accumulate gradients.
                    must be factor of batch_size
    """
    if batch_size % sub_batch_size != 0:
        raise RuntimeError(f'batch_size = {batch_size} must be disivible by '
                f'sub_batch_size = {sub_batch_size}')

    client = Client('aiayn')
    client.init_pubsub(pubsub_project_id, 'aiayn')
    client.clear()
    grid_map = dict(loss = (0,0,1,1))
    client.set_layout(grid_map)

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
        en_range = (en_lengths.min().item(), en_lengths.max().item())
        de_range = (de_lengths.min().item(), de_lengths.max().item())
        if en_range[1] > 150 or de_range[1] > 150:
            print(f'Skipping: en = {en_range}, de = {de_range}')
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
        client.tandem_lines('loss', step, [loss], fig_kwargs={'width':1800})
        if step % report_every == 0:
            print(f', loss = {loss:5.4f}', flush=True)

        if step % ckpt_every == 0 and step > 0 and step != resume_ckpt:
            path = ckpt_templ.format(step)
            print(f'Saving {path}', flush=True)
            run.save(path)

if __name__ == '__main__':
    fire.Fire(main)


