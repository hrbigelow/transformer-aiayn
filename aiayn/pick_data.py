import fire
import numpy as np
import tensorflow as tf
from aiayn import data

def main(token_info_file, data_path, num_samples, swap_source_target):
    token_info = np.load(token_info_file)
    dataset = data.main_dataset(data_path, token_info, 100, num_samples,
            swap_source_target, 1000)
    it = iter(dataset)
    encs, decs, _, _ = next(iter(dataset))

    special_toks = { 
            token_info['eos'].item(): '<EOS>',
            token_info['bos'].item(): '<BOS>',
            token_info['mask'].item(): '<MASK>'
            }
    encs = data.de_tokenize(encs, special_toks)
    decs = data.de_tokenize(decs, special_toks)
    for enc, dec in zip(encs, decs):
        print(f'{enc}\n{dec}\n\n')

if __name__ == '__main__':
    fire.Fire(main)
    




