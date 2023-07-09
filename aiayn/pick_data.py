import fire
from aiayn import data

def main(data_path, num_samples):
    dataset, ds_info = data.base_dataset(data_path, 'train', 2)
    dataset = data.pipe_dataset(dataset, ds_info, 500, num_samples)
    it = iter(dataset)
    enc_input, dec_input, _, _ = next(it)
    encs = data.de_tokenize(enc_input)
    decs = data.de_tokenize(dec_input)

    for enc, dec in zip(encs, decs):
        print(f'{enc=} => {dec=}')

if __name__ == '__main__':
    fire.Fire(main)
    




