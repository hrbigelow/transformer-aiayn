import os
import sys
import fire
import numpy as np
from collections import Counter
from transformers import GPT2TokenizerFast
import tensorflow as tf
import tensorflow_datasets as tfds

def de_tokenize(tokens):
    """
    tokens: np array of shape batch, length
    returns: python list of strings
    """
    tok = get_tokenizer()
    ans = []
    special_toks = (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
    for i in range(tokens.shape[0]):
        toks = tokens[i]
        toks = np.extract(~np.isin(toks, special_toks), toks) 
        toks = tok.convert_ids_to_tokens(toks)
        text = tok.convert_tokens_to_string(toks)
        ans.append(text)
    return ans

def token_dataset(download_dir, split, dataset_name, nproc):
    tokenizer = get_tokenizer()
    builder = tfds.builder(f'wmt14_translate/{dataset_name}', data_dir=download_dir)
    builder.download_and_prepare(download_dir=download_dir)
    ds = builder.as_dataset(split=split, shuffle_files=True)
    ds_info = builder.info

    def tokenize_fn(item):
        def _py_fn(one, two):
            one = tokenizer(one.numpy().decode())['input_ids']
            two = tokenizer(two.numpy().decode())['input_ids']
            return tf.constant(one, dtype=tf.uint16), tf.constant(two, dtype=tf.uint16)
        return tf.py_function(_py_fn, inp=item.values(), Tout=[tf.uint16, tf.uint16])

    ds = ds.map(tokenize_fn, num_parallel_calls=nproc, deterministic=False)
    print('prefetching...')
    ds.prefetch(ds_info.splits[split].num_examples)

    ds = ds.cache(f'{download_dir}/{dataset_name}-cache')
    # iterate once to populate cache
    return ds, ds_info

def pad_dataset(token_ds, shuffle_size, max_sentence_length):
    tokenizer = get_tokenizer()
    bos = tf.constant([tokenizer.bos_token_id], tf.uint16)
    eos = tf.constant([tokenizer.eos_token_id], tf.uint16)
    pad_id = tokenizer.pad_token_id

    def pad_tokens_fn(tok1, tok2):
        tok1 = tf.cast(tok1, tf.uint16)
        tok2 = tf.cast(tok2, tf.uint16)
        sen_len1 = tf.shape(tok1)[0]
        sen_len2 = tf.shape(tok2)[0]
        l1 = max_sentence_length - sen_len1 - 2
        l2 = max_sentence_length - sen_len2 - 2
        pad1 = tf.cast(tf.fill((l1,), pad_id), dtype=tf.uint16)
        pad2 = tf.cast(tf.fill((l2,), pad_id), dtype=tf.uint16)
        tok1 = tf.concat(values=(bos, tok1, eos, pad1), axis=0)
        tok2 = tf.concat(values=(bos, tok2, eos, pad2), axis=0)
        return tok1, tok2, sen_len1, sen_len2 

    def maxlen_fn(tok1, tok2):
        return tf.maximum(tf.shape(tok1)[0], tf.shape(tok2)[0]) <= max_sentence_length - 2

    ds = token_ds.filter(maxlen_fn)
    ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
    ds = ds.map(pad_tokens_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds

def pipe_dataset(pad_ds, batch_size, swap_source_target):
    ds = pad_ds.repeat()
    def swap_fn(t1, t2, s1, s2):
        return t2, t1, s2, s1

    if swap_source_target:
        ds = ds.map(swap_fn)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = tfds.as_numpy(ds)
    return ds

def main_dataset(data_path, max_sentence_length, batch_size, swap_source_target):
    token_ds = tf.data.Dataset.load(data_path)
    shuffle_size = len(token_ds)
    pad_ds = pad_dataset(token_ds, shuffle_size, max_sentence_length)
    return pipe_dataset(pad_ds, batch_size, swap_source_target)

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


