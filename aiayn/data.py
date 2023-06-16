import os
import sys
import fire
import numpy as np
from collections import Counter
from transformers import GPT2TokenizerFast
import tensorflow as tf
import tensorflow_datasets as tfds

def base_dataset(download_dir, split, nproc):
    tokenizer = get_tokenizer()
    builder = tfds.builder('wmt14_translate/de-en')
    builder.download_and_prepare(download_dir=download_dir)
    ds = builder.as_dataset(split=split, shuffle_files=True)
    ds_info = builder.info

    def tokenize_fn(item):
        def _py_fn(en, de):
            en = tokenizer(en.numpy().decode())['input_ids']
            de = tokenizer(de.numpy().decode())['input_ids']
            return tf.constant(en), tf.constant(de)
        return tf.py_function(_py_fn, inp=item.values(), Tout=[tf.int32, tf.int32])

    ds = ds.map(tokenize_fn, num_parallel_calls=nproc, deterministic=False)
    # ds.prefetch(ds_info.splits[split].num_examples)
    ds = ds.cache(f'{download_dir}/wmt14_cache')
    # iterate once to populate cache
    return ds, ds_info

def pipe_dataset(dataset, ds_info, max_sentence_length, batch_size):
    tokenizer = get_tokenizer()

    def maxlen_fn(tok1, tok2):
        return tf.maximum(tf.shape(tok1)[0], tf.shape(tok2)[0]) <= max_sentence_length - 2

    bos = tf.constant([tokenizer.bos_token_id])
    eos = tf.constant([tokenizer.eos_token_id])
    pad_id = tokenizer.pad_token_id

    def pad_tokens_fn(tok1, tok2):
        l1 = max_sentence_length - tf.shape(tok1)[0] - 2
        l2 = max_sentence_length - tf.shape(tok2)[0] - 2
        pad1 = tf.fill((l1,), pad_id)
        pad2 = tf.fill((l2,), pad_id)
        tok1 = tf.concat(values=(bos, tok1, eos, pad1), axis=0)
        tok2 = tf.concat(values=(bos, tok2, eos, pad2), axis=0)
        return tok1, tok2 

    ds = dataset.filter(maxlen_fn)
    ds = ds.map(pad_tokens_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    # ds = ds.shuffle(ds_info.splits['train'].num_examples)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = tfds.as_numpy(ds)
    return ds

def token_histo(dataset):
    """
    Compute the token histograms for each column
    """
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)
    histo1 = tf.zeros((vocab_size,), dtype=tf.int32)
    histo2 = tf.zeros((vocab_size,), dtype=tf.int32)

    def histo_fn(histos, toks):
        h1, h2 = histos
        t1, t2 = toks
        h1 = tf.tensor_scatter_nd_add(h1, tf.expand_dims(t1, -1), tf.ones_like(t1))
        h2 = tf.tensor_scatter_nd_add(h2, tf.expand_dims(t2, -1), tf.ones_like(t2))
        return h1, h2
    return dataset.reduce((histo1, histo2), histo_fn)

def save_token_info(dataset, data_dir):
    tz = get_tokenizer()
    print('got tokenizer')
    cts1, cts2 = token_histo(dataset)
    print('created token histos')
    cts1 = tf.cast(cts1, tf.float32)
    cts2 = tf.cast(cts2, tf.float32)
    h1 = cts1 / tf.reduce_sum(cts1)
    h2 = cts2 / tf.reduce_sum(cts2)
    np.savez(f'{data_dir}/token_info.npz', 
            en=h1.numpy(), de=h2.numpy(),
            pad_token_id=tz.pad_token_id,
            bos_token_id=tz.bos_token_id,
            eos_token_id=tz.eos_token_id)

def prepare(data_dir, split):
    if not os.access(data_dir, os.W_OK):
        raise RuntimeError(f'Couldn\'t open {data_dir=} for writing')
    ds, ds_info = base_dataset(data_dir, split, 2)
    for i, _ in enumerate(iter(ds)):
        if i % 10000 == 0:
            print(f'Sample {i} iterated')
    print(f'Finished')

def load_token_info(data_dir):
    path = f'{data_dir}/token_info.npz'
    return np.load(path)

def column_counts(dataset, column):
    def map_fn(examples, accu):
        accu += Counter(examples)
    cts = Counter()
    dataset.map(map_fn, batched=True, input_columns=column, fn_kwargs=dict(accu=cts))
    return dict(cts)

BOS = '<|BOS|>'
EOS = '<|EOS|>'
PAD = '<|PAD|>'

def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_tokens([BOS, EOS, PAD])
    tokenizer.pad_token = PAD
    tokenizer.bos_token = BOS
    tokenizer.eos_token = EOS
    return tokenizer

if __name__ == '__main__':
    cmds = dict(prepare=prepare, save_token_info=save_token_info)
    fire.Fire(cmds)


