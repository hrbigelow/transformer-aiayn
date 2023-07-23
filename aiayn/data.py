import tensorflow as tf
import tensorflow_datasets as tfds
import os
import sys
import fire
import numpy as np
import jax.numpy as jnp
from collections import Counter, deque
from transformers import GPT2TokenizerFast
# import seqio
import functools
# from seqio.feature_converters import EncDecFeatureConverter


def prepare(registry_item, tfrec_glob):
    seqio.TaskRegistry.remove(registry_item)
    seqio.TaskRegistry.add(
            registry_item,
            seqio.dataset_providers.TFExampleDataSource(
                { 'train': tfrec_glob },
                { 'x': tf.io.FixedLenFeature([], tf.string), 
                  'y': tf.io.FixedLenFeature([], tf.string) }
                ),
            preprocessors = [ 
                convert_tfrec_dataset,
                functools.partial(
                    seqio.preprocessors.apply_feature_converter,
                    sequence_length={'x': 80, 'y': 128},
                    feature_converter=EncDecFeatureConverter(pack=True)
                    )
                ],
            output_features = { 
                'inputs': seqio.Feature(
                    seqio.PassThroughVocabulary(50257),
                    add_eos=False,
                    dtype=tf.int32,
                    rank=1
                    ),
                'targets': seqio.Feature(
                    seqio.PassThroughVocabulary(50257),
                    add_eos=False,
                    dtype=tf.int32,
                    rank=1
                    )
                }
            )

def test_prepare(registry_item):
    seqio.TaskRegistry.remove(registry_item)
    seqio.TaskRegistry.add(
            registry_item,
            source=seqio.TfdsDataSource('wmt14_translate/de-en:1.0.0'),
            output_features= {}
            )
        
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

def write_records(ds, record_templ, num_shards, shards=None):
    options = tf.io.TFRecordOptions(
            compression_type=None,
            input_buffer_size=1000000,
            output_buffer_size=1000000)

    if shards is None:
        shards = range(num_shards)

    for shard in shards:
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

def reorder(ds, bufsize, xmax):
    # it = ds.as_numpy_iterator()
    spec = tf.TensorSpec(shape=(None,), dtype=tf.int32)
     # spec = tf.data.DatasetSpec.from_value(ds)
    def filt_fn(x):
        return tf.size(x) <= xmax

    ds = ds.filter(filt_fn)
    it = iter(ds)

    def gen():
        dq = deque()
        while True:
            if len(dq) < bufsize:
                x = next(it)
                dq.append(x)
            yield dq.popleft()
    ds = ds.from_generator(gen, output_signature = spec)
    return ds

def pad_and_filter(ds, feature_lengths, pad_value):
    # filter out any entries longer than those given in feature_lengths
    # pad
    def filter_fn(ent):
        return tf.logical_and(
                tf.size(ent['inputs']) <= feature_lengths['inputs'],
                tf.size(ent['targets']) <= feature_lengths['targets'])

    def pad_fn(ent):
        lengths = { k: tf.size(v) for k, v in ent.items() }
        padded = {
                k: tf.ensure_shape(
                    tf.concat(
                    values=[v, tf.fill([feature_lengths[k] - tf.size(v)], pad_value)], 
                    axis=0),
                    [feature_lengths[k]]
                    )

                for k, v in ent.items()
                }
        record = dict(
                inputs=(padded['inputs'], lengths['inputs']),
                targets=(padded['targets'], lengths['targets'])
                )
        return record

    return ds.filter(filter_fn).map(pad_fn)

def condense_buffer(buf_sz, used, buf):
    """
    buf: tensor of 1 or more dimensions
    buf_sz: number of entries considered present in the buffer
    used: 1d boolean tensor, marking entries in axis 0 
    returns: buffer with unused slices packed towards zero
             and new buf_sz
    """
    used = tf.pad(used, tf.stack([[0, buf_sz - tf.size(used)]]), constant_values = False)
    unused_inds = tf.where(tf.logical_not(used))
    retained_entries = tf.gather_nd(buf, unused_inds, 0)
    new_buf_sz = tf.shape(retained_entries)[0]
    buf = tf.scatter_nd(tf.reshape(tf.range(new_buf_sz), (-1,1)), retained_entries,
            tf.shape(buf))
    return buf

def masked_sizes(pad_ds, batch_sz, num_tries, feat_lengths):
    """
    pad_ds: dataset from pad_and_filter 
    Maintain a buffer of token sequences that have been consumed from the source,
    but not yet yielded.  Parallel to this, maintains the buffer of corresponding
    token sequence lengths.  Both buffers are batch_sz * 2.

    """
    tf_update_fn = tf.tensor_scatter_nd_update
    input_len = feat_lengths['inputs']
    target_len = feat_lengths['targets']
    batch_ds = pad_ds.batch(batch_sz)
    ds_iter = iter(batch_ds)

    def refill_buffer(buf_sz, buf, more):
        dest = tf.reshape(tf.range(buf_sz, buf_sz + batch_sz), (-1,1))
        # tf.print(f'{buf_sz=}, {buf.shape=}, {dest.shape=} {more.shape=}')
        return tf_update_fn(buf, dest, more)

    def test_fit(pack, buf, cap):
        return tf.less_equal(pack + buf[:batch_sz], cap)

    def gen():
        # capacity
        cap = dict(
            inputs=tf.fill(batch_sz, input_len),
            targets=tf.fill(batch_sz, target_len))
        # tokens buffer
        toks = dict(
                inputs=tf.zeros((batch_sz * 2, input_len), dtype=tf.int32),
                targets=tf.zeros((batch_sz * 2, target_len), dtype=tf.int32))
        # lengths buffers
        lens = dict(
                inputs=tf.zeros((batch_sz * 2,), dtype=tf.int32),
                targets=tf.zeros((batch_sz * 2,), dtype=tf.int32))

        # the high-water-mark for tok_buf1 and slen_buf1
        # will be the same value for each field, but works well with nest.map_structure
        buf_sz = 0
        while True:
            pack = dict(
                inputs=tf.zeros(batch_sz, dtype=tf.int32),
                targets=tf.zeros(batch_sz, dtype=tf.int32))

            for i in range(num_tries):
                if buf_sz < batch_sz:
                    batch = next(ds_iter)
                    new_toks = { 'inputs': batch['inputs'][0], 'targets': batch['targets'][0] }
                    new_lens = { 'inputs': batch['inputs'][1], 'targets': batch['targets'][1] }
                    fn = functools.partial(refill_buffer, buf_sz)
                    toks = tf.nest.map_structure(fn, toks, new_toks)
                    lens = tf.nest.map_structure(fn, lens, new_lens)
                    buf_sz += batch_sz
                
                # tf.print('buf_sz, batch_sz: ', buf_sz, batch_sz)
                fits = tf.nest.map_structure(test_fit, pack, lens, cap)
                fits = tf.logical_and(*fits.values())
                fits_inds = tf.where(fits)
                fits_fn = lambda l: tf.where(fits, l[:batch_sz], tf.constant([0]))
                try_sz = tf.nest.map_structure(fits_fn, lens)

                y = dict(
                        inputs=(toks['inputs'][:batch_sz], try_sz['inputs']),
                        targets=(toks['targets'][:batch_sz], try_sz['targets']))

                yield y

                fn = functools.partial(condense_buffer, buf_sz, fits)
                toks = tf.nest.map_structure(fn, toks)
                lens = tf.nest.map_structure(fn, lens)
                buf_sz -= tf.reduce_sum(tf.cast(fits, tf.int32))
                pack = tf.nest.map_structure(tf.add, pack, try_sz)

    spec = dict(
            inputs=(
                tf.TensorSpec(shape=(batch_sz, input_len), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_sz,), dtype=tf.int32)),
            targets=(
                tf.TensorSpec(shape=(batch_sz, target_len), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_sz,), dtype=tf.int32))
            )

    size_ds = tf.data.Dataset.from_generator(gen, output_signature=spec)
    # return size_ds
    def transpose_fn(item):
        inp, trg = item['inputs'], item['targets']
        return dict( 
                inputs=(
                    tf.transpose(inp[0], (1,0,2)),
                    tf.transpose(inp[1], (1,0))),
                targets=(
                    tf.transpose(trg[0], (1,0,2)),
                    tf.transpose(trg[1], (1,0)))
                )
    return size_ds.batch(num_tries).map(transpose_fn)

@tf.function
def pack_sequences(seq_and_len):
    """
    b = batch, p = tries, o = output sequence
    seqs: bpo
    seq_lens: pb 
    out_len: scalar
    returns: po

    indices[slice,coord]
    updates[slice,elem] = RANDOM(0, 10, FLOAT)
    output[dest,elem] = 0.0 
    output[indices[slice,:],elem] = updates[slice,elem]

    slice:  B,O,P
    coord:  2
    elem:   () 
    dest:   B,O

    indices: B,O,P,2 (slice, coord)
    output: B,O  (dest, elem)
    updates: B,O,P  (slice, elem)
    """
    seqs, seq_lens = seq_and_len
    B = tf.shape(seqs)[0]
    P = tf.shape(seqs)[1]
    O = tf.shape(seqs)[2]

    # seqs = tf.transpose(seqs, (1,0,2))
    # seq_lens = tf.transpose(seq_lens, (1,0))

    seq_lens = tf.expand_dims(seq_lens, 2)
    o_rang = tf.expand_dims(tf.expand_dims(tf.range(O), 0), 1)
    b_rang = tf.expand_dims(tf.expand_dims(tf.range(B), 1), 2)
    p_rang = tf.expand_dims(tf.expand_dims(tf.range(P), 0), 2)
    seq_begs = tf.cumsum(seq_lens, axis=1, exclusive=True)
    seq_ends = tf.cumsum(seq_lens, axis=1)
    pre_inds = tf.add(seq_begs, o_rang)
    mask = tf.greater(seq_lens, o_rang) 
    inds = tf.where(mask, pre_inds, O) # B,P,O
    slice_shape = tf.shape(inds)
    new_inds = tf.stack([
                tf.broadcast_to(b_rang, slice_shape), 
                inds,
                ], axis=3)
    ids = tf.broadcast_to(p_rang, slice_shape)
    packed_seqs = tf.scatter_nd(new_inds, seqs, shape=(B,O+1))[:,:O]
    # return new_inds, ids, seqs
    out_shape = (B,O+1)
    packed_ids = tf.tensor_scatter_nd_update(tf.fill(out_shape, -1), new_inds, ids)[:,:O]
    packed_cts = tf.reduce_sum(tf.cast(tf.not_equal(packed_ids, -1), tf.int32), axis=1)
    return packed_seqs, packed_ids, packed_cts

def pack_dataset(toks_ds, batch_size, num_tries, feature_lengths, pad_value):
    """
    toks_ds:  dataset yielding { 'input': input_tokens, 'target': target_tokens }
    batch_size:  internal size for batching the packing operation
    num_tries:  number of internal iterations for trying to pack a batch
    feature_lengths: map of { 'input': max_input_len, 'target': max_target_len } 
    """
    pad_ds = pad_and_filter(toks_ds, feature_lengths, pad_value)
    sz_ds = masked_sizes(pad_ds, batch_size, num_tries, feature_lengths)
    def pack_pair_fn(item):
        return dict(
                inputs=pack_sequences(item['inputs']),
                targets=pack_sequences(item['targets'])
                )

    pack_ds = sz_ds.map(pack_pair_fn)
    return pack_ds.unbatch()

def pack_dataset_bck(ds, xmax, ymax, max_tries, threshold, max_queue_size, pad_value):
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
        # if total_len > max_len: 
            # raise RuntimeError(f'Error: {total_len=} > {max_len=}')

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

    def gen():
        dq = deque()
        xpack = tf.zeros(xmax)
        xmask = tf.zeros(xmax)
        ypack = tf.zeros(ymax)
        ymask = tf.zeros(ymax)
        while True:
            fill_queue(dq, it)
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
            # print(f'packing after {i} tries')
            # xpack, xmask = pack(xs, xmax - x_remain, xmax)
            # ypack, ymask = pack(ys, ymax - y_remain, ymax)
            yield xpack, ypack, xmask, ymask, xmax - x_remain, ymax - y_remain, i

    """
    def gen():
        dq = deque()
        while True:
            fill_queue(dq, it)
            yield make_pack(dq)
    """

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


