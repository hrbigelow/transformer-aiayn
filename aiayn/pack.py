import tensorflow as tf
from dataclasses import dataclass
import functools

"""
Utility for end-to-end packing of variable-length token sequence pairs for
encoder-decoder language models.

Usage: Given a dataset token_ds yielding pairs of variable-length token sequence pairs:
    { 'inputs': input_token_seq, 'targets': target_token_seq }

pack_ds = pack.pack_dataset(token_ds, feature_lengths) 
with spec:
    { 'inputs':  (packed_input_seqs, packed_input_ids, total_pack_length),
      'targets': (packed_input_seqs, packed_input_ids, total_pack_length) }

"""
def pad_and_filter(ds, feature_lengths, pad_value):
    """
    (This implementation is the same speed as the original)
    ds:  dataset with elements 
         { 'inputs': (rank-1 int32 tensor of varying length),
           'targets': (rank-1 int32 tensor of varying length) }
         meant to be unpadded (but with special tokens added) tokenized
         sentence pairs.  (despite the plural 'inputs' and 'targets', each
         dataset item is a single sentence pair.

    feature_lengths: { 'inputs': <max_input_length>,
                       'targets': <max_target_length> }
         the maximum desired input and target lengths.
    
    output: a dataset with structure:
         { 'inputs': (padded_input, orig_length),
           'targets': (padded_target, orig_length) }
         
    """
    def filter_fn(ent):
        fn = lambda ten, L: tf.less(tf.size(ten), L)
        res = tf.nest.map_structure(fn, ent, feature_lengths)
        return tf.logical_and(*res.values())

    def pad_fn(ent):
        def pad_len_fn(ten, L):
            return (tf.concat([ten, tf.fill([L-tf.size(ten)], pad_value)], 0), 
                    tf.size(ten))
        return tf.nest.map_structure(pad_len_fn, ent, feature_lengths)

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

    def transpose_fn(item):
        return dict( 
                inputs=(
                    tf.transpose(item['inputs'][0], (1,0,2)),
                    tf.transpose(item['inputs'][1], (1,0))),
                targets=(
                    tf.transpose(item['targets'][0], (1,0,2)),
                    tf.transpose(item['targets'][1], (1,0)))
                )
    return size_ds.batch(num_tries).map(transpose_fn)

@tf.function
def get_scatter_inds(batch_inds, seqs, lens):
    """
    batch_inds: bpo, a materialized tensor holding value b at index bpo
    seqs: bpo
    lens: bp

    returns: bpo2  last slice is [B,O] coordinates
    """
    O = tf.shape(seqs)[2]
    lens = lens[:, :, None]
    o_rang = tf.range(O)[None, None, :]
    begs = tf.cumsum(lens, axis=1, exclusive=True)
    pre_inds = tf.add(begs, o_rang)
    mask = tf.greater(lens, o_rang) 
    inds = tf.where(mask, pre_inds, O) # B,P,O
    slice_shape = tf.shape(inds)
    return tf.stack([batch_inds, inds], axis=3)

@tf.function
def pack_values(scatter_inds, values):
    """
    scatter_inds: bpo2
    values: bpo
    dest: bo

    Performs the operation:
    dest[*inds[b,p,o]] = values[b,p,o]
    """
    B = tf.shape(values)[0]
    O = tf.shape(values)[2]
    dest = tf.fill((B,O+1), -1)
    return tf.tensor_scatter_nd_update(dest, scatter_inds, values)[:,:O]
def pack_sequences(seq_and_len, pfx):
    seqs, lens = seq_and_len
    B = tf.shape(seqs)[0]
    P = tf.shape(seqs)[1]
    O = tf.shape(seqs)[2]
    gbatch = tf.broadcast_to(tf.range(B)[:,None,None], (B,P,O))
    seqids = tf.broadcast_to(tf.range(P)[None,:,None], (B,P,O))
    tokids = tf.broadcast_to(tf.range(O)[None,None,:], (B,P,O))
    scatter_inds = get_scatter_inds(gbatch, seqs, lens)
    seqs = pack_values(scatter_inds, seqs)
    seqids = pack_values(scatter_inds, seqids)
    tokids = pack_values(scatter_inds, tokids)
    counts = tf.reduce_sum(tf.cast(tf.not_equal(seqids, -1), tf.int32), axis=1)
    keys = tuple(f'{pfx}_{sfx}' for sfx in ('seqs', 'seqids', 'tokids', 'counts'))
    return dict(zip(keys, (seqs, seqids, tokids, counts)))

def pack_dataset(toks_ds, feature_lengths, batch_size=1000, num_tries=10, pad_value=-1):
    """
    toks_ds:  dataset with spec { 'input': input_tokens, 'target': target_tokens }
    feature_lengths: map of { 'input': max_input_len, 'target': max_target_len } 
    batch_size:  internal size for batching the packing operation
    num_tries:  number of internal iterations for trying to pack a batch
    """
    pad_ds = pad_and_filter(toks_ds, feature_lengths, pad_value)
    sz_ds = masked_sizes(pad_ds, batch_size, num_tries, feature_lengths)

    def pack_pair_fn(item):
        inputs = pack_sequences(item['inputs'], 'inputs') 
        targets = pack_sequences(item['targets'], 'targets')
        return { **inputs, **targets }

    pack_ds = sz_ds.map(pack_pair_fn)
    return pack_ds.unbatch()


# OLD

@dataclass
class SeqLen:
    seqs: tf.Tensor # bpo
    lens: tf.Tensor # bp

def pack_tries(tries_ds, feature_lengths, bp_shape):
    # This is horribly slow for some reason
    B,P = bp_shape
    O = max(feature_lengths.values())
    gbatch = tf.broadcast_to(tf.range(B)[:,None,None], (B,P,O))
    seqids = tf.broadcast_to(tf.range(P)[None,:,None], (B,P,O))
    tokids = tf.broadcast_to(tf.range(O)[None,None,:], (B,P,O))
    
    gbatch = { k: gbatch[:,:,:v] for k, v in feature_lengths.items() }
    seqids = { k: seqids[:,:,:v] for k, v in feature_lengths.items() }
    tokids = { k: tokids[:,:,:v] for k, v in feature_lengths.items() }
    
    def map_fn(elem):
        def pack_fn(cell, gbatch, seqids, tokids): 
            scatter_inds = get_scatter_inds(gbatch, cell.seqs, cell.lens)
            fn = functools.partial(pack_values, scatter_inds)
            vals = { 'seqs': cell.seqs, 'sids': seqids, 'tids': tokids }
            pack = tf.nest.map_structure(fn, vals)
            cts = tf.reduce_sum(tf.cast(tf.not_equal(pack['seqs'], -1), tf.int32), axis=1)
            return { **pack, 'cts': cts }
        elem = {k: SeqLen(*v) for k, v in elem.items()}
        return tf.nest.map_structure(pack_fn, elem, gbatch, seqids, tokids)
    return tries_ds.map(map_fn)

