import fire
import numpy as np
import tensorflow as tf
from aiayn import data

def main(data_dir, data_glob, num_samples, swap_pairs, seed):
    data.set_config(data_dir=data_dir)
    data.set_config(tokenizer='gpt2_tokenizer')
    toks_ds = data.load_tfrecord_dataset(data_glob, swap_pairs)
    toks_ds = toks_ds.shuffle(100000, seed)
    it = toks_ds.as_numpy_iterator()

    for _ in range(num_samples):
        item = next(it) 
        encs = data.de_tokenize(item['inputs'][None,:], {})
        decs = data.de_tokenize(item['targets'][None,:], {})
        for enc, dec in zip(encs, decs):
            print(f'{enc}\n{dec}\n\n')

if __name__ == '__main__':
    fire.Fire(main)
    




