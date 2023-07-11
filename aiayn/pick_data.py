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

    encs = data.de_tokenize(encs, True)
    decs = data.de_tokenize(decs, True)
    for enc, dec in zip(encs, decs):
        print(f'{enc}\n{dec}\n\n')

if __name__ == '__main__':
    fire.Fire(main)
    




