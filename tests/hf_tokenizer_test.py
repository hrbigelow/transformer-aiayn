"""
Experiments to see if I can reproduce the AIAYN tokenizer workflow using HuggingFace
"""
import fire
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import datasets
import pdb

def map_fn(item):
    def _py_fn(one, two):
        return one.decode('utf-8'), two.decode('utf-8')
    return tf.numpy_function(_py_fn, inp=item.values(), Tout=[tf.string, tf.string])

def train_tokenizer(dataset_name, split, data_dir, vocab_size, out_file):
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare()
    ds = builder.as_dataset(split=split, shuffle_files=False)
    num_elems = len(ds) * 2
    # ds = ds.map(map_fn)
    ds = ds.batch(1000)

    def convert(ds):
        it = ds.as_numpy_iterator()
        decode = np.vectorize(lambda x: x.decode())
        while True:
            item = next(it, None)
            if item is None:
                return
            one, two = item.values()
            yield decode(one)
            yield decode(two)

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

    trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=['[UNK]', '[PAD]', '[EOS]', '[BOS]'])

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(convert(ds), trainer, num_elems)
    tokenizer.save(out_file)


def load_tokenizer(file):
    return Tokenizer.from_file(file)


if __name__ == '__main__':
    fire.Fire(train_tokenizer)


