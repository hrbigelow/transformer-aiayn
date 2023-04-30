import fire
import numpy as np
from . import model
from . import data
from . import state
import torch
from torch.optim import Adam

def main(batch_size, bin_size, data_path, ckpt_path=None):
    run = state.Run(model=model.Model, data=data.Data, opt=) 
    ds = data.get_dataset(data_path)
    token_histo = data.load_token_histo(data_path)
    tokenizer = data.get_tokenizer()
    print(f'Prepared dataset')

    hps = model.HyperParams()
    hps.T = len(tokenizer)

    print(f'Instantiating run')
    if ckpt_path is None:
        run.init(hps)
    else:
        run.load(ckpt_path)

    # mod = torch.compile(mod)
    loss = run.model.Loss(token_histo, tokenizer.pad_token_id)
    opt = Adam(run.model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    run.sched.update(0)

    inds_gen = run.data.batched_sample(batch_size, bin_size, ds.num_rows)
    for epoch, step, inds in inds_gen:
        enc_input, dec_input, en_lengths, de_lengths = data.make_batch(ds, inds, 
                tokenizer.pad_token_id)
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
        xent = loss(dec_input, dec_output)
        run.model.zero_grad()
        xent.backward()
        run.opt.step()
        run.sched.update(step)
        print(f', loss = {xent.item():5.4f}')

        if step % 1000 == 0 and step > 0:

if __name__ == '__main__':
    fire.Fire(main)


