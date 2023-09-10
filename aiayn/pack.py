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
    { 'inputs': { 'seqs': ..., 'seqids': ..., 'tokids': ..., 'counts': ... }, 
      'targets': <same as inputs> }

    seqs: o: concatenated token sequences
    seqids: o: integer id denoting which source sequence the token belongs
    tokids: o: integer id denoting the position in the source sequence 
    counts: p: total length of sequence p in the pack, which is in [0, num_tries)

    for non-existent tokens, seqs will have `pad_value`, while seqids and tokids will
    have -1.

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
            ten = tf.cast(ten, tf.int32)
            return dict(
                    toks=tf.concat([ten, tf.fill([L-tf.size(ten)], pad_value)], 0), 
                    lens=tf.size(ten))
        return tf.nest.map_structure(pad_len_fn, ent, feature_lengths)

    return ds.filter(filter_fn).map(pad_fn)


def make_gen(ds_iter, batch_sz, num_tries, input_len, target_len):

    def get_condense_inds(used):
        """
        used: b
        """
        used = tf.concat((used, tf.fill(batch_sz, False)), axis=0)
        copy = tf.where(tf.logical_not(used))[:,0]
        fill = tf.constant(2*batch_sz-1, shape=2*batch_sz, dtype=tf.int64)
        return tf.concat((copy, fill), axis=0)[:2*batch_sz,None]

    def condense(inds, buf):
        return tf.gather_nd(buf, inds)

    def refill_buffer(buf_sz, buf, more):
        # buf[*inds[b],...] = more[b,...]
        this_batch_sz = tf.shape(more)[0]
        inds = tf.range(buf_sz, buf_sz + this_batch_sz)[:,None]
        return tf.tensor_scatter_nd_update(buf, inds, more)

    capacity = dict(
            inputs=tf.fill((batch_sz,), input_len),
            targets=tf.fill((batch_sz,), target_len))

    seqs_buf = dict(
            inputs = {
                'toks': tf.zeros((batch_sz * 2, input_len), tf.int32),
                'lens': tf.zeros((batch_sz * 2), tf.int32) },
            targets = {
                'toks': tf.zeros((batch_sz * 2, target_len), tf.int32),
                'lens': tf.zeros((batch_sz * 2), tf.int32) }
            )

    def gen():
        # the high-water-mark for tok_buf1 and slen_buf1
        buf_sz = tf.constant(0) 
        more_input = True
        i = 0
        nonlocal seqs_buf
        nonlocal capacity

        # The i != 0 clause guarantees that a full set of tries will be yielded 
        while buf_sz != 0 or more_input or i != 0:
            if i == 0:
                pack = dict(
                    inputs=tf.zeros((batch_sz,), tf.int32),
                    targets=tf.zeros((batch_sz,), tf.int32))
            i = (i + 1) % num_tries

            if buf_sz < tf.constant(batch_sz):
                new_seqs = next(ds_iter, None)
                if new_seqs is None:
                    more_input = False
                else:
                    new_batch_sz = tf.shape(new_seqs['inputs']['toks'])[0]
                    fn = functools.partial(refill_buffer, buf_sz)
                    seqs_buf = tf.nest.map_structure(fn, seqs_buf, new_seqs)
                    buf_sz = tf.add(buf_sz, new_batch_sz)
            
            # tf.print('buf_sz, batch_sz: ', buf_sz, batch_sz)
            lens = { k: v['lens'] for k, v in seqs_buf.items() }
            fits_fn = lambda pack, buf, cap: tf.less_equal(pack + buf[:batch_sz], cap)
            fits = tf.nest.map_structure(fits_fn, pack, lens, capacity)
            fits = tf.logical_and(*fits.values()) # True if both input and target fit

            # exclude sequences that don't exist in the buffer (are beyond buf_sz)
            fits = tf.logical_and(fits, tf.less(tf.range(batch_sz), buf_sz))
            fits_fn = lambda l: tf.where(fits, l[:batch_sz], tf.constant([0]))
            try_sz = tf.nest.map_structure(fits_fn, lens)

            # We assume buf_sz >= batch_sz
            yield dict(
                    inputs=(seqs_buf['inputs']['toks'][:batch_sz,:], try_sz['inputs']),
                    targets=(seqs_buf['targets']['toks'][:batch_sz,:], try_sz['targets']))

            condense_inds = get_condense_inds(fits)
            condense_fn = functools.partial(condense, condense_inds)
            seqs_buf = tf.nest.map_structure(condense_fn, seqs_buf)

            fits_active = tf.cast(fits, tf.int32)
            # print(f'{buf_sz=}, {fits=}') 
            buf_sz = tf.subtract(buf_sz, tf.reduce_sum(tf.cast(fits, tf.int32)))
            pack = tf.nest.map_structure(tf.add, pack, try_sz)
            # print(f'{buf_sz=} {more_input=}')

    spec = dict(
            inputs=(
                tf.TensorSpec(shape=(batch_sz, input_len), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_sz,), dtype=tf.int32)),
            targets=(
                tf.TensorSpec(shape=(batch_sz, target_len), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_sz,), dtype=tf.int32))
            )

    return tf.data.Dataset.from_generator(gen, output_signature=spec)

def masked_sizes(pad_ds, batch_sz, num_tries, feat_lengths):
    """
    pad_ds: dataset from pad_and_filter 
    Maintain a buffer of token sequences that have been consumed from the source,
    but not yet yielded.  Parallel to this, maintains the buffer of corresponding
    token sequence lengths.  Both buffers are batch_sz * 2.
    """
    input_len = feat_lengths['inputs']
    target_len = feat_lengths['targets']
    batch_ds = pad_ds.batch(batch_sz)
    ds_iter = iter(batch_ds)
    size_ds = make_gen(ds_iter, batch_sz, num_tries, input_len, target_len)
    return size_ds.batch(num_tries)

@tf.function
def get_scatter_inds(batch_inds, seqs, lens):
    """
    batch_inds: pbo, a materialized tensor holding value b at index pbo
    seqs: pbo
    lens: pb

    returns: pbo2  last slice is [B,O] coordinates
    """
    O = tf.shape(seqs)[2]
    lens = lens[:, :, None]
    o_rang = tf.range(O)[None, None, :]
    begs = tf.cumsum(lens, axis=0, exclusive=True)
    pre_inds = tf.add(begs, o_rang)
    mask = tf.greater(lens, o_rang) 
    inds = tf.where(mask, pre_inds, O) # B,P,O
    slice_shape = tf.shape(inds)
    return tf.stack([batch_inds, inds], axis=3)

@tf.function
def pack_values(scatter_inds, pad_value, values):
    """
    scatter_inds: pbo2
    values: pbo
    dest: bo

    Performs the operation:
    dest[*inds[p,b,o]] = values[p,b,o]
    """
    B = tf.shape(values)[1]
    O = tf.shape(values)[2]
    dest = tf.fill((B,O+1), pad_value)
    return tf.tensor_scatter_nd_update(dest, scatter_inds, values)[:,:O]

@tf.function
def pack_sequences(seq_and_len, pad_value):
    """
    seqs: pbo
    lens: pb 

    output: dict of
    seqs:   bo
    seqids: bo
    tokids: bo
    counts: bp
    
    """
    seqs, lens = seq_and_len
    P = tf.shape(seqs)[0]
    B = tf.shape(seqs)[1]
    O = tf.shape(seqs)[2]
    seqids = tf.broadcast_to(tf.range(P)[:,None,None], (P,B,O))
    gbatch = tf.broadcast_to(tf.range(B)[None,:,None], (P,B,O))
    tokids = tf.broadcast_to(tf.range(O)[None,None,:], (P,B,O))
    scatter_inds = get_scatter_inds(gbatch, seqs, lens)

    pack_seqs_fn = functools.partial(pack_values, scatter_inds, pad_value)
    pack_ids_fn = functools.partial(pack_values, scatter_inds, -1)
    seqs = tf.nest.map_structure(pack_seqs_fn, seqs)
    seqids, tokids = tf.nest.map_structure(pack_ids_fn, (seqids, tokids))
    counts = tf.transpose(lens, (1,0))
    return dict(seqs=seqs, seqids=seqids, tokids=tokids, counts=counts)


def pack_dataset(toks_ds, feature_lengths, packing_batch=1000, num_tries=10, pad_value=-1):
    """
    Packs items in toks_ds into blocks given by `feature_lengths`
    toks_ds:  dataset with spec { 'input': input_tokens, 'target': target_tokens }
    feature_lengths: map of { 'input': max_input_len, 'target': max_target_len } 
    packing_batch:  internal size for batching the packing operation
    num_tries:  number of internal iterations for trying to pack a batch

    The final structure of the returned dataset is 
    """
    pad_ds = pad_and_filter(toks_ds, feature_lengths, pad_value)
    sz_ds = masked_sizes(pad_ds, packing_batch, num_tries, feature_lengths)

    def pack_pair_fn(item):
        inputs = pack_sequences(item['inputs'], pad_value) 
        targets = pack_sequences(item['targets'], pad_value)
        return dict(inputs=inputs, targets=targets)
    
    def filter_fn(item):
        return tf.not_equal(item['inputs']['tokids'][0], -1)

    return sz_ds.map(pack_pair_fn).unbatch().filter(filter_fn)

def empty_pack(length, num_tries, pad_value):
    pad = tf.cast(tf.fill(length, pad_value), tf.int32)
    null = tf.cast(tf.fill(length, -1), tf.int32)
    return dict(
            seqs=pad,
            seqids=null,
            tokids=null,
            counts=tf.zeros(num_tries, dtype=tf.int32)
            )

def filler_dataset(feature_lengths, dataset_size, num_tries, pad_value=-1):
    item = { feature: empty_pack(length, num_tries, pad_value) 
            for feature, length in feature_lengths.items() }
    return tf.data.Dataset.from_tensors(item).repeat(dataset_size)


def unpack_dataset(pack_ds, num_tries, pad_value, process_batch_size=10):
    """
    pack_ds: a packed dataset (as returned by `pack_dataset`) (unbatched)
    num_tries: the parameter given to pack_dataset
    pad_value: parameter given to pack_dataset
    returns:  toks_ds (as received by `pack_dataset`)
    
    Performs the operation:
    dest[*inds[b,p,o]] = seqs[b,p,o]

    indices[slice,coord] = RANDOM(0, DIMS(dest)[coord], INT)
    updates[slice,elem] = RANDOM(0, 10, FLOAT)
    output[dest,elem] = 0.0
    output[indices[slice,:],elem] = updates[slice,elem]

    In this case:

    elem: ()
    slice: B,O
    dest: B,P,O

    indices: B,O,3
    updates: B,O
    output: B,P,O
    """

    P = num_tries

    def unpack_fn(cell):

        seqs = cell['seqs']
        seqids = cell['seqids']
        tokids = cell['tokids']
        counts = cell['counts'] # B
        B = tf.shape(seqs)[0]
        O = tf.shape(seqs)[1]

        # Elements in the input which are padding will be scattered to (b,0,O)
        # and then trimmed out
        gbatch = tf.broadcast_to(tf.range(B)[:,None], (B,O))
        tokids = tf.where(tf.not_equal(tokids, -1), tokids, O)
        seqids = tf.where(tf.not_equal(seqids, -1), seqids, 0)
        inds = tf.stack([gbatch, seqids, tokids], axis=2)
        dest = tf.fill((B,P,O+1), pad_value)
        unpack = tf.tensor_scatter_nd_update(dest, inds, seqs)[:,:,:O]
        return unpack, counts

    def unpack_map_fn(item):
        return dict(
                inputs=unpack_fn(item['inputs']), 
                targets=unpack_fn(item['targets']))

    def filt_fn(item):
        _, input_len = item['inputs']
        _, target_len = item['targets']
        return tf.logical_or(
                tf.not_equal(input_len, 0),
                tf.not_equal(target_len, 0)
                )

    def trim_map_fn(item):
        input_seq, input_len = item['inputs']
        target_seq, target_len = item['targets']
        return dict(
                inputs=input_seq[:input_len],
                targets=target_seq[:target_len]
                )

    return (
            pack_ds.batch(process_batch_size)
            .map(unpack_map_fn)
            .unbatch() 
            .unbatch()
            .filter(filt_fn)
            .map(trim_map_fn)
            )

