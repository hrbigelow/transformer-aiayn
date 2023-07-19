import os
import sys
import fire
import numpy as np
import jax.numpy as jnp
from collections import Counter
from transformers import GPT2TokenizerFast
import tensorflow as tf
import tensorflow_datasets as tfds


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

    def swap_fn(t1, t2):
        return t2, t1

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

    ds = token_ds
    if swap_pairs:
        ds = ds.map(swap_fn)

    ds = ds.filter(maxlen_fn)
    ds = ds.shuffle(shuffle_size, rng_key, reshuffle_each_iteration=True)
    ds = ds.map(pad_tokens_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds

def pipe_dataset(pad_ds, batch_size, initial_step=0):
    ds = pad_ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.skip(initial_step)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = tfds.as_numpy(ds)
    return ds

def main_dataset(data_name, token_info, max_sentence_length, batch_size, 
        swap_source_target, rng_key, initial_step, shuffle_size=None):
    data_dir = CONFIG.get('data_dir', None)
    if data_dir is None:
        raise RuntimeError('Please call data.set_config first')
    token_ds = tf.data.Dataset.load(os.path.join(data_dir, data_name))

    if shuffle_size is None:
        shuffle_size = len(token_ds)
    pad_ds = pad_dataset(token_ds, token_info, shuffle_size, swap_source_target, 
            max_sentence_length, rng_key)
    return pipe_dataset(pad_ds, batch_size, initial_step)

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

def column_counts(dataset, column):
    def map_fn(examples, accu):
        accu += Counter(examples)
    cts = Counter()
    dataset.map(map_fn, batched=True, input_columns=column, fn_kwargs=dict(accu=cts))
    return dict(cts)

def get_tokenizer():
    path = os.path.join(CONFIG['data_dir'], CONFIG['tokenizer'], 'tokenizer.json')
    from transformers import PreTrainedTokenizerFast as ptf
    return ptf(tokenizer_file=path)

if __name__ == '__main__':
    cmds = dict(save_token_info=save_token_info)
    fire.Fire(cmds)


