import fire
import numpy as np
import torch as t
from collections import defaultdict, Counter
import datasets
from transformers import GPT2TokenizerFast

def to_tokens(examples, tokenizer):
    ret = dict()
    for col, text in examples.items():
        toks = [tokenizer(s)['input_ids'] for s in text]
        ret.update({ f'{col}_tok': toks, f'{col}_length': list(map(len, toks)) })
    return ret

def column_counts(dataset, column):
    def map_fn(examples, accu):
        accu += Counter(examples)
    cts = Counter()
    dataset.map(map_fn, batched=True, input_columns=column, fn_kwargs=dict(accu=cts))
    return dict(cts)

def compute_binmap(counts, num_bins):
    total = sum(counts.values())
    target_bin_size = total / num_bins
    binmap = {} # length -> bin
    bins = []
    bin_counts = 0
    bin_start = 0
    length_bin = 0
    for length in sorted(counts.keys()):
        if bin_counts >= target_bin_size:
            binmap.update({ sz: length_bin for sz in range(bin_start, length) })
            bin_counts = 0
            bin_start = length
            length_bin += 1
        bin_counts += counts[length]
    if bin_counts != 0:
        binmap.update({ sz: length_bin for sz in range(bin_start, length) })

    return binmap

def assign_bin(tok_lengths, binmap, out_column):
    """
    binmap: Dict[length, bin]
    """
    return { out_column: list(map(binmap.get, tok_lengths)) } 

def shuffle_inner(ary, block_size, rng):
    """
    Shuffle each contiguous subrange of size block_size, preserving order of the
    blocks themselves.
    Example: np.arange(10), block_size = 5 -> [4,1,3,2,0,9,6,5,7,8]
    """
    assert isinstance(ary, np.ndarray)
    for beg in range(0, len(ary), block_size):
        rng.shuffle(ary[beg:beg+block_size])
    return ary

def shuffle_outer(ary, block_size, rng):
    """
    Shuffle the order of blocks, preserving order of entries in each block.
    ary.shape[0] must be divisible by block_size
    Example: np.arange(12), block_size = 3 -> [3,4,5,0,1,2,9,10,11,6,7,8]
    """
    assert isinstance(ary, np.ndarray)
    n = ary.shape[0]
    if n % block_size != 0:
        raise RuntimeError(
            f'shuffle_outer: ary.shape[0] = {n} not divisible by '
            f'block_size = {block_size}')
    perm = np.arange(0, n, block_size)
    rng.shuffle(perm)
    inds = perm.repeat(block_size) + np.arange(n) % block_size
    shuf = np.take(ary, inds)
    return shuf

def preprocess(data_path, num_proc=8):
    """
    """
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = datasets.load_dataset('wmt14', 'de-en', split='train')
    ds = ds.flatten().rename_columns(
            { 'translation.de': 'de', 'translation.en': 'en'})
    
    # ds = ds.shard(100, 1)
    ds = ds.map(to_tokens, batched=True, 
            fn_kwargs=dict(tokenizer=tokenizer),
            num_proc=num_proc)
    # en_counts = column_counts(ds, 'en_length')
    # en_binmap = compute_binmap(en_counts, 100) 
    # ds = ds.map(assign_bin, batched=True, 
            # input_columns='en_length',
            # fn_kwargs=dict(binmap=en_binmap, out_column='en_bin'), 
            # num_proc=num_proc)
    ds = ds.remove_columns(('en', 'de'))
    ds = ds.sort('en_length')
    ds.set_format('torch')
    # return ds, tokenizer
    # print(ds.features)
    print(f'Saving dataset to {data_path}.  Columns: {ds.column_names}')
    ds.save_to_disk(data_path)


def get_dataset(data_path):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return datasets.load_from_disk(data_path), tokenizer

def inverse_mod(val, modulo):
    # amount needed to round up to nearest modulo
    return - (val % -modulo)

def batched_sample(rng, batch_size, bin_size, dataset_size):
    """
    Infinitely generates batch_size batches of indices in range(dataset_size)
    uniformly, even if dataset_size % batch_size != 0
    """
    offset = 0
    while True:
        main_inds = np.arange(offset, dataset_size)
        extra = inverse_mod(main_inds.shape[0], batch_size) 
        wrap_inds = np.arange(extra) # wrapping around to the next epoch
        inds = np.concatenate((wrap_inds, main_inds))
        assert inds.shape[0] % batch_size == 0
        offset = extra
        inds = shuffle_inner(inds, bin_size, rng)
        inds = shuffle_outer(inds, batch_size, rng)
        for b in range(0, inds.shape[0], batch_size):
            yield inds[b:b+batch_size]

def make_batch(dataset, inds, pad_value):
    """
    """
    entries = dataset[inds]
    def _padded_stack(tensors, pad_value):
        max_len = max(ten.shape[0] for ten in tensors)
        st = t.full((len(tensors), max_len), pad_value, dtype=t.int32)
        for i, ten in enumerate(tensors):
            st[i,:ten.shape[0]] = ten 
        return st

    en = _padded_stack(entries['en_tok'], pad_value)
    de = _padded_stack(entries['de_tok'], pad_value)
    return en, de


if __name__ == '__main__':
    fire.Fire(preprocess)


