import fire
import numpy as np
from . import model
from . import data
import torch
from torch.optim import Adam

def main(batch_size, bin_size, data_path, seed=0):
    ds = data.get_dataset(data_path)
    token_histo = data.load_token_histo(data_path)
    tokenizer = data.get_tokenizer()
    print(f'Prepared dataset')

    hps = model.HyperParams()
    hps.T = len(tokenizer)
    mod = model.Model(hps)
    print(f'Instantiated model')
    # mod = torch.compile(mod)
    loss = model.Loss(token_histo, tokenizer.pad_token_id)
    opt = Adam(mod.parameters(), betas=(0.9, 0.98), eps=1e-9)
    rng = np.random.mtrand.RandomState(seed)
    lr_sched = model.CustomScheduler(opt, hps)
    lr_sched.update(0)

    inds_gen = data.batched_sample(rng, batch_size, bin_size, ds.num_rows)
    for epoch, step, inds in inds_gen:
        enc_input, dec_input, en_lengths, de_lengths = data.make_batch(ds, inds, 
                tokenizer.pad_token_id)
        dec_output = mod(enc_input, dec_input)
        xent = loss(dec_input, dec_output)
        mod.zero_grad()
        xent.backward()
        opt.step()
        lr_sched.update(step)
        lr = lr_sched.current_lr()
        en_rng = (en_lengths.min().item(), en_lengths.max().item())
        de_rng = (de_lengths.min().item(), de_lengths.max().item())
        print(f'epoch = {epoch}, step = {step}, '
                f'en = {en_rng}, de = {de_rng}, '
                f'lr = {lr:7.6f}, loss = {xent.item():5.4f}')

if __name__ == '__main__':
    fire.Fire(main)


