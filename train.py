import fire
import numpy as np
import model
import data
from torch.optim import Adam

def main(batch_size, bin_size, data_path, seed=0):
    ds = data.get_dataset(data_path)
    token_histo = data.load_token_histo(data_path)
    tokenizer = data.get_tokenizer()

    dims = model.Dims()
    dims.T = tokenizer.vocab_size
    mod = model.Model(dims)
    loss = model.Loss(token_histo)
    opt = Adam(mod.parameters(), betas=(0.9, 0.98), eps=1e-9)
    rng = np.random.mtrand.RandomState(seed)

    inds_gen = data.batched_sample(rng, batch_size, bin_size, ds.num_rows)
    for step, inds in enumerate(inds_gen):
        enc_input, dec_input = data.make_batch(ds, inds, tokenizer.pad_token_id)
        dec_output = mod(enc_input, dec_input)
        xent = loss(dec_input, dec_output)
        mod.zero_grad()
        xent.backward()
        opt.step()
        print(f'step = {step}, loss = {xent.item():5.4f}')

if __name__ == '__main__':
    fire.Fire(main)


