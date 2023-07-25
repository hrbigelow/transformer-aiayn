import tensorflow as tf
import tensorflow_datasets as tfds
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

def convert_tfrec_dataset(tfrec_dataset, swap_inputs_targets):
    return tfrec_dataset.map(
            functools.partial(parse_record, swap_inputs_targets),
            num_parallel_calls=tf.data.AUTOTUNE)

CONFIG = { }

def set_config(**kwargs):
    global CONFIG 
    CONFIG.update(kwargs)

def tokenize(query):
    tokenizer = get_tokenizer()
    return jnp.array(tokenizer(query)['input_ids'])

def get_special_tokens(token_info):
    return { 
            token_info['eos'].item(): '<EOS>',
            token_info['bos'].item(): '<BOS>',
            token_info['mask'].item(): '<MASK>'
            }

def de_tokenize(tokens, special_toks={}):
    """
    tokens: np array of shape batch, length
    special_toks: map of id => token_string
    returns: python list of strings
    """
    tz = get_tokenizer()
    idmap = { v: k for k, v in tz.vocab.items() }
    idmap.update(special_toks)
    ans = []
    for i in range(tokens.shape[0]):
        tokids = tokens[i]
        toks = [idmap[el.item()] for el in tokids]
        # text = ''.join(idmap[el] for el in toks)
        # toks = tok.convert_ids_to_tokens(toks)
        text = tz.convert_tokens_to_string(toks)
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


def pad_dataset(token_ds, token_info, shuffle_size, swap_pairs, max_sentence_length,
        rng_key):
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

    def expand_fn(rec):
        return rec['x'], rec['y']

    def swap_fn(x, y):
        return y, x

    def pad_tokens_fn(x, y):
        # x = tf.sparse.to_dense(rec['x']) # This was needed when using 
        # VarLenFeature during parse_record
        # y = tf.sparse.to_dense(rec['y'])
        x = tf.cast(x, tf.uint16)
        y = tf.cast(y, tf.uint16)
        xlen = tf.size(x)
        ylen = tf.size(y)
        l1 = max_sentence_length - xlen 
        l2 = max_sentence_length - ylen - 2
        mask1 = tf.cast(tf.fill((l1,), mask_id), dtype=tf.uint16)
        mask2 = tf.cast(tf.fill((l2,), mask_id), dtype=tf.uint16)
        x = tf.concat(values=(x, mask1), axis=0)
        y = tf.concat(values=(bos, y, eos, mask2), axis=0)
        return x, y, xlen, ylen 

    def maxlen_fn(rec):
        x, y = rec['x'], rec['y']
        return tf.maximum(tf.size(x), tf.size(y)) <= max_sentence_length - 2

    ds = token_ds
    ds = ds.filter(maxlen_fn)
    ds = ds.map(expand_fn)
    if swap_pairs:
        ds = ds.map(swap_fn)
    ds = ds.shuffle(shuffle_size, rng_key, reshuffle_each_iteration=True)
    ds = ds.map(pad_tokens_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds

def main_dataset(tfrecord_glob, bos_id, eos_id, max_len1, max_len2, max_tries,
        pack_threshold, batch_size, swap_source_target, seed, initial_step,
        shuffle_size=10000):
    ds = load_tfrecord_dataset(tfrecord_glob) 
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_size, seed, reshuffle_each_iteration=True)
    ds = add_special_tokens(ds, swap_source_target, bos_id, eos_id)
    ds = pack_dataset(ds, max_len1, max_len2, max_tries=max_tries,
            threshold=pack_threshold, max_queue_size=100, pad_value=-1)
    ds = ds.prefetch(1000)
    ds = ds.batch(batch_size)
    return ds

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

def load_token_info(token_file_name):
    data_dir = CONFIG.get('data_dir', None)
    if data_dir is None: 
        raise RuntimeError(f'Please call data.set_data_dir first')
    path = os.path.join(data_dir, token_file_name)
    try:
        return np.load(path)
    except:
        raise RuntimeError(f'Could not load token info file {path}')

def get_tokenizer():
    path = os.path.join(CONFIG['data_dir'], CONFIG['tokenizer'], 'tokenizer.json')
    from transformers import PreTrainedTokenizerFast as ptf
    return ptf(tokenizer_file=path)

if __name__ == '__main__':
    cmds = dict(save_token_info=save_token_info)
    fire.Fire(cmds)

