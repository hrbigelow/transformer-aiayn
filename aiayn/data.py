import os
import sys
import fire
import numpy as np
import jax.numpy as jnp
from collections import Counter
from transformers import GPT2TokenizerFast
import tensorflow as tf
import tensorflow_datasets as tfds

def tokenize(query):
    tokenizer = get_tokenizer()
    return jnp.array(tokenizer(query)['input_ids'])

def de_tokenize(tokens, special_toks=None):
    """
    tokens: np array of shape batch, length
    returns: python list of strings
    """
    tok = get_tokenizer()
    ans = []
    for i in range(tokens.shape[0]):
        toks = tokens[i]
        if special_toks is not None:
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

    # ds = ds.cache(f'{download_dir}/{dataset_name}-cache')
    # iterate once to populate cache
    return ds, ds_info

def pad_dataset(token_ds, token_info, shuffle_size, max_sentence_length):
    """
    Appends padding to first sentence
    Appends eos + padding to second sentence
    token_info: loaded from save_token_info output 
    """

    # this is done so that the model will learn that 'eos' only leads to 'eos'.
    eos_id = token_info['eos'].item()
    bos_id = token_info['bos'].item()
    mask_id = token_info['mask'].item()
    bos = tf.constant([bos_id], tf.uint16)
    eos = tf.constant([eos_id, eos_id], tf.uint16)

    def pad_tokens_fn(tok1, tok2):
        tok1 = tf.cast(tok1, tf.uint16)
        tok2 = tf.cast(tok2, tf.uint16)
        sen_len1 = tf.shape(tok1)[0]
        sen_len2 = tf.shape(tok2)[0]
        l1 = max_sentence_length - sen_len1
        l2 = max_sentence_length - sen_len2 - 2
        mask1 = tf.cast(tf.fill((l1,), mask_id), dtype=tf.uint16)
        mask2 = tf.cast(tf.fill((l2,), mask_id), dtype=tf.uint16)
        tok1 = tf.concat(values=(tok1, mask1), axis=0)
        tok2 = tf.concat(values=(bos, tok2, eos, mask2), axis=0)
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

def main_dataset(data_path, token_info, max_sentence_length, batch_size, 
        swap_source_target, shuffle_size=None):
    token_ds = tf.data.Dataset.load(data_path)
    if shuffle_size is None:
        shuffle_size = len(token_ds)
    pad_ds = pad_dataset(token_ds, token_info, shuffle_size, max_sentence_length)
    return pipe_dataset(pad_ds, batch_size, swap_source_target)

def token_histo(token_ds, column_num):
    """
    Compute the token histograms for each co
    token_ds:  the token dataset from data.token_dataset 
    column_num: 0 or 1 to designate which of the pair of sentences desired
    """
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)
    histo = tf.zeros((vocab_size,), dtype=tf.int32)
    def histo_fn(h, toks):
        tok = tf.cast(toks[column_num], tf.int32)
        h = tf.tensor_scatter_nd_add(h, tf.expand_dims(tok, -1), tf.ones_like(tok))
        return h 
    return token_ds.reduce(histo, histo_fn)

def save_token_info(token_ds_path, column_num, histo_path):
    """
    token_ds_path: path to saved token dataset
    column_num: 0 or 1 to designate which sentence
    histo_path: path to output file 
    """
    token_ds = tf.data.Dataset.load(token_ds_path)
    cts = token_histo(token_ds, column_num)
    cts = tf.cast(cts, tf.float32)
    h = cts / tf.reduce_sum(cts)
    bos_id = h.shape[0]
    eos_id = h.shape[0] + 1 
    mask_id = h.shape[0] + 2  
    h = np.concatenate((h, np.zeros(2)))
    np.savez(histo_path, histo=h.numpy(), bos=bos_id, eos=eos_id, mask=mask_id)

def column_counts(dataset, column):
    def map_fn(examples, accu):
        accu += Counter(examples)
    cts = Counter()
    dataset.map(map_fn, batched=True, input_columns=column, fn_kwargs=dict(accu=cts))
    return dict(cts)

def get_tokenizer():
    return GPT2TokenizerFast.from_pretrained('gpt2')

if __name__ == '__main__':
    cmds = dict(save_token_info=save_token_info)
    fire.Fire(cmds)


