import fire
import numpy as np
import tensorflow as tf
from aiayn import data
from transformers import PreTrainedTokenizerFast

def tsv_output(data_glob, tokenizer_file, swap_pairs):
    """
    Prepare tsv output with fields `id`, `input`, `target`
    """
    tz = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file) 
    def t2s(toks):
        return tz.convert_tokens_to_string(tz.convert_ids_to_tokens(toks, True))

    toks_ds = data.load_tfrecord_dataset(data_glob, swap_pairs)
    it = toks_ds.as_numpy_iterator()
    for _id, item in enumerate(it):
        input_seq = t2s(item['inputs'])
        target_seq = t2s(item['targets'])
        print(f'{_id}\t{input_seq}\t{target_seq}')


def pick(data_dir, data_glob, num_samples, swap_pairs, seed):
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
    opts = dict(batch=tsv_output, pick=pick)
    fire.Fire(opts)


