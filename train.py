import fire
import numpy as np
import model
import data
from torch.optim import Adam

def main(batch_size, bin_size, data_path, seed=0):
    ds, tokenizer = data.get_dataset(data_path)
    tokenizer.pad_token = '<|PAD|>'

    dims = model.Dims()
    dims.T = tokenizer.vocab_size
    mod = model.Model(dims)
    opt = Adam(mod.parameters(), betas=(0.9, 0.98), eps=1e-9)
    rng = np.random.mtrand.RandomState(seed)

    inds_gen = data.batched_sample(rng, batch_size, bin_size, ds.num_rows)
    for inds in inds_gen: 
        en, de = data.make_batch(ds, inds, tokenizer.pad_token_id)
        print(en.shape, de.shape)


if __name__ == '__main__':
    fire.Fire(main)


