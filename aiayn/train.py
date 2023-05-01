import fire
import numpy as np
from . import model
from . import data
from . import state
import torch
from torch.optim import Adam

def main(batch_size, data_path, ckpt_templ, resume_ckpt=None, compile_model=False):
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

    inds_gen = iter(run.data)
    for epoch, step, inds in inds_gen:
        enc_input, dec_input, en_lengths, de_lengths = data.make_batch(ds, inds, 
                tokenizer.pad_token_id, 'cuda')
        en_range = (en_lengths.min().item(), en_lengths.max().item())
        de_range = (de_lengths.min().item(), de_lengths.max().item())
        lr = run.sched.current_lr()
        print(f'epoch = {epoch}, step = {step}, '
                f'en = {en_range}, de = {de_range}, '
                f'lr = {lr:7.6f}', end='', flush=True) 
        if en_range[1] > 150:
            print(f'Skipping too-long sentence to avoid OOM')
            continue
        dec_output = run.model(enc_input, dec_input)
        xent = run.model.loss(dec_input, dec_output)
        run.model.zero_grad()
        xent.backward()
        run.opt.step()
        run.sched.update(step)
        print(f', loss = {xent.item():5.4f}')

        if step % 1000 == 0 and step > 0 and step != resume_ckpt:
            path = ckpt_templ.format(step)
            print(f'Saving {path}')
            run.save(path)

if __name__ == '__main__':
    fire.Fire(main)


