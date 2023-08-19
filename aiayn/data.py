from collections import namedtuple
import tensorflow as tf
import tensorflow_datasets as tfds
from tokenizers import Tokenizer
import os
import sys
import fire
import numpy as np
import functools
import jax.numpy as jnp
from aiayn import pack

def parse_example(swap, example):
    schema = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
            }
    record = tf.io.parse_single_example(example, schema)
    return parse_record(swap, record)

def parse_record(swap, record):
    # see https://colab.research.google.com/notebooks/tpu.ipynb#scrollTo=LtAVr-4CP1rp&line=26&uniqifier=1
    # print(f'example[x]: {example["x"]}')
    x = tf.io.parse_tensor(record['x'], out_type=tf.uint16)
    y = tf.io.parse_tensor(record['y'], out_type=tf.uint16)
    x = tf.cast(x, tf.int32)
    y = tf.cast(y, tf.int32)
    if swap:
        x, y = y, x
    # tf.print(x)
    return { 'inputs': x, 'targets': y }

CONFIG = { }

def set_config(**kwargs):
    global CONFIG 
    CONFIG.update(kwargs)

def tokenize(tokenizer, queries, pad_value):
    """
    queries
    """
    toks = tokenizer(queries)['input_ids']
    max_len = max(len(l) for l in toks)
    toks = [ l + [pad_value] * (max_len - len(l)) for l in toks]
    return jnp.array(toks, dtype=jnp.int32)


def get_special_tokens(token_info):
    return { 
            token_info['eos'].item(): '<EOS>',
            token_info['bos'].item(): '<BOS>',
            token_info['mask'].item(): '<MASK>'
            }

def de_tokenize(tokenizer, tokens, eos_id, special_toks={}):
    """
    tokenizer: must have eos_token_id set
    tokens: np array of shape [batch, length]
    special_toks: map of id => token_string.  If set, additionally decode these 
    returns: python list of strings
    """
    idmap = { v: k for k, v in tokenizer.vocab.items() }
    idmap.update(special_toks)
    ans = []

    for i in range(tokens.shape[0]):
        tokids = tokens[i].tolist()
        end = len(tokids) if eos_id not in tokids else tokids.index(eos_id)
        toks = [idmap[el] for el in tokids[:end] if el in idmap]
        # text = ''.join(idmap[el] for el in toks)
        # toks = tok.convert_ids_to_tokens(toks)
        text = tokenizer.convert_tokens_to_string(toks)
        ans.append(text)
    return ans

def load_tfrecord_dataset(tfrecord_glob, swap_inputs_targets):
    # optimization advice from https://codelabs.developers.google.com/codelabs/keras-flowers-data#4
    filenames = tf.io.gfile.glob(tfrecord_glob)
    if len(filenames) == 0:
        raise RuntimeError(
                f'load_tfrecord_dataset: '
                f'Couldn\'t find any files in tfrecord_glob pattern \'{tfrecord_glob}\'')

    AUTOTUNE = tf.data.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    fn = functools.partial(parse_example, swap_inputs_targets)
    return dataset.map(fn, num_parallel_calls=AUTOTUNE)

def add_special_tokens(token_ds, bos_id, eos_id):
    """
    Appends padding to first sentence
    Appends eos + padding to second sentence
    token_info: loaded from save_token_info output 
    """

    # this is done so that the model will learn that 'eos' only leads to 'eos'.
    bos = tf.constant([bos_id], tf.uint16)
    eos = tf.constant([eos_id, eos_id], tf.uint16)

    def toks_fn(item):
        x = tf.cast(item['inputs'], tf.uint16)
        y = tf.cast(item['targets'], tf.uint16)
        return dict(
                inputs=x,
                targets=tf.concat(values=(bos, y, eos), axis=0))

    return token_ds.map(toks_fn, num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

def length_histo(token_ds):
    histo = tf.zeros((1000,2), dtype=tf.int32)
    def histo_fn(h, toks):
        l1 = tf.minimum(tf.size(toks[0]), 999)
        l2 = tf.minimum(tf.size(toks[1]), 999)
        inds = tf.reshape(tf.stack([l1, 0, l2, 1]), (2, 2))
        return tf.tensor_scatter_nd_add(h, inds, tf.ones([2], dtype=tf.int32))
    return token_ds.reduce(histo, histo_fn)

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

def load_token_info(token_path):
    try:
        with tf.io.gfile.GFile(token_path, 'rb') as file:
            z = np.load(file)
            return dict(
                    bos=z['bos'].item(),
                    eos=z['eos'].item(),
                    mask=z['mask'].item(),
                    n_vocab=z['histo'].shape[0],
                    histo=z['histo'])
    except:
        raise RuntimeError(f'Could not load token info file {path}')

def get_tokenizer(tokenizer_path):
    try:
        content = tf.io.gfile.GFile(tokenizer_path).read()
    except Exception as ex:
        raise RuntimeError(
            f'Couldn\'t load tokenizer JSON file from {tokenizer_path}')
    return Tokenizer.from_str(content)

def get_special_tokens(tokenizer_path):
    import json
    try:
        j = json.load(tf.io.gfile.GFile(tokenizer_path))
    except Exception as ex:
        raise RuntimeError(
            f'Couldn\'t load tokenizer JSON file from {tokenizer_path}')
    Toks = namedtuple('tokens', 'bos_id eos_id pad_id')
    m = { item['content']: item['id'] for item in j['added_tokens'] }
    return Toks(m['[BOS]'], m['[EOS]'], m['[PAD]'])

if __name__ == '__main__':
    cmds = dict(save_token_info=save_token_info)
    fire.Fire(cmds)

