import os
import sys
import fire
import numpy as np
import jax.numpy as jnp
from collections import Counter, deque
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

def write_records(ds, record_templ, num_shards):
    options = tf.io.TFRecordOptions(
            compression_type=None,
            input_buffer_size=1000000,
            output_buffer_size=1000000)

    for shard in range(num_shards):
        record_path = record_templ.format(shard) 
        ds_shard = ds.shard(num_shards, shard)
        with tf.io.TFRecordWriter(record_path, options) as file_writer:
            for t1, t2 in iter(ds_shard):
                s1 = tf.io.serialize_tensor(t1)
                s2 = tf.io.serialize_tensor(t2)
                b1 = tf.train.BytesList(value=[s1.numpy()])
                b2 = tf.train.BytesList(value=[s2.numpy()])

                record_bytes = tf.train.Example(
                    features=tf.train.Features(feature={
                        'x': tf.train.Feature(bytes_list=b1),
                        'y': tf.train.Feature(bytes_list=b2)
                        }
                    )
                ).SerializeToString()
                file_writer.write(record_bytes)
        print(f'Wrote {record_path} of {num_shards}')

def parse_record(example):
    # see https://colab.research.google.com/notebooks/tpu.ipynb#scrollTo=LtAVr-4CP1rp&line=26&uniqifier=1
    schema = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
            }
    example = tf.io.parse_single_example(example, schema)
    return {
            'x': tf.io.parse_tensor(example['x'], out_type=tf.uint16),
            'y': tf.io.parse_tensor(example['y'], out_type=tf.uint16)
            }

def load_tfrecord_dataset(tfrecord_glob):
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
    return dataset.map(parse_record, num_parallel_calls=AUTOTUNE)

def pack_dataset(ds, xmax, ymax, max_tries, threshold, max_queue_size, pad_value):
    """
    ds:  dataset producing x, y token seqs, unpadded but with special tokens 
    xmax, ymax: maximum length for packed x (y) sequence
    max_tries:  maximum number of searches through the queue to extend a pack
    threshold: [0, 1]:  mark a pack complete if it attains this fullness threshold
    max_queue_size:  maximum size of deque used as a look-ahead buffer
    pad_value:  use this value to pad the returned packs

    Returns:
    packx, packy, maskx, masky, lenx, leny, num_retries
    packx, packy:  concatenated token sequences, padded to xmax (ymax) with pad_value
    maskx, masky:  tensors with values 0, 1, etc identifying the entries in packx
                   (packy) or -1 if non-sample
    lenx, leny:    number of sample tokens in packx (packy)
    num_retries:    number of queue items tried to extend this pack
    """
    it = ds.as_numpy_iterator()

    def pack(tensors, total_len, max_len):
        # assert sum(tf.size(t) for t in tensors) == total_len
        if total_len > max_len: 
            raise RuntimeError(f'Error: {total_len=} > {max_len=}')

        pad = tf.fill(max_len - total_len, pad_value)
        idx_tensors = [tf.fill(tf.size(ten), i) for i, ten in enumerate(tensors)]
        return (tf.concat(tensors + [pad], axis=0), 
                tf.concat(idx_tensors + [pad], axis=0))

    def fill_queue(dq, it):
        while len(dq) < max_queue_size:
            x, y = next(it)
            if tf.size(x).numpy() > xmax or tf.size(y).numpy() > ymax:
                continue
            dq.append((x, y))

    def make_pack(dq):
        xs, ys = [], []
        x_remain, y_remain = xmax, ymax 
        max_empty = int((1.0 - threshold) * (xmax + ymax))
        for i in range(max_tries):
            if x_remain + y_remain < max_empty:
                break
            x, y = dq.popleft()
            xil = tf.size(x).numpy()
            yil = tf.size(y).numpy()
            if xil > x_remain or yil > y_remain:
                # recycle this item
                dq.append((x, y))
            else:
                xs.append(x)
                ys.append(y)
                x_remain -= xil
                y_remain -= yil
        xpack, xmask = pack(xs, xmax - x_remain, xmax)
        ypack, ymask = pack(ys, ymax - y_remain, ymax)
        return xpack, ypack, xmask, ymask, xmax - x_remain, ymax - y_remain, i

    def gen():
        dq = deque()
        while True:
            fill_queue(dq, it)
            yield make_pack(dq)

    return ds.from_generator(gen, 
            output_signature = (
                tf.TensorSpec(shape=(xmax,), dtype=tf.int32),
                tf.TensorSpec(shape=(ymax,), dtype=tf.int32),
                tf.TensorSpec(shape=(xmax,), dtype=tf.int32),
                tf.TensorSpec(shape=(ymax,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            )

def add_special_tokens(token_ds, swap_pairs, bos_id, eos_id):
    """
    Appends padding to first sentence
    Appends eos + padding to second sentence
    token_info: loaded from save_token_info output 
    """

    # this is done so that the model will learn that 'eos' only leads to 'eos'.
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
        y = tf.concat(values=(bos, y, eos), axis=0)
        return x, y

    ds = token_ds
    ds = ds.map(expand_fn)
    if swap_pairs:
        ds = ds.map(swap_fn)
    ds = ds.map(pad_tokens_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds


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

def pipe_dataset(ds, seed, shuffle_size, batch_size, initial_step=0):
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_size, seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    # ds = ds.skip(initial_step)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def main_dataset(tfrecord_glob, bos_id, eos_id, max_len1, max_len2, max_tries,
        pack_threshold, batch_size, swap_source_target, rng_key, initial_step,
        shuffle_size=10000):
    token_ds = load_tfrecord_dataset(tfrecord_glob) 
    ds = add_special_tokens(token_ds, swap_source_target, bos_id, eos_id)
    ds = pack_dataset(ds, max_len1, max_len2, max_tries=max_tries,
            threshold=pack_threshold, max_queue_size=100, pad_value=-1)
    return pipe_dataset(ds, rng_key, shuffle_size, batch_size, initial_step)

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


