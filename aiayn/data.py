import fire
import numpy as np
import torch as t
from collections import defaultdict, Counter
import datasets
from transformers import GPT2TokenizerFast
from . import state


def token_histo(dataset, column):
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    # get histogram of tokens for token column
    def map_fn(toks, cts):
        all_toks = t.cat(toks)
        cts.scatter_add_(0, all_toks, t.ones_like(all_toks))

    cts = t.zeros(vocab_size, dtype=t.int64)
    kwargs = dict(cts=cts)
    # dataset = dataset.shard(100, 1)
    dataset.map(map_fn, batched=True, input_columns=column, fn_kwargs=kwargs)
            # with_rank=True, num_proc=num_proc)
    return cts

def column_counts(dataset, column):
    def map_fn(examples, accu):
        accu += Counter(examples)
    cts = Counter()
    dataset.map(map_fn, batched=True, input_columns=column, fn_kwargs=dict(accu=cts))
    return dict(cts)

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

def tokenize_data(dataset, num_proc):
    ds = dataset.flatten().rename_columns(
            { 'translation.de': 'de', 'translation.en': 'en'})
    
    def map_fn(examples, tokenizer):
        ret = dict()
        bos_ten = t.tensor([tokenizer.vocab[BOS]])
        eos_ten = t.tensor([tokenizer.vocab[EOS]])
        for col, text in examples.items():
            toks = []
            for s in text:
                tok = tokenizer(s, return_tensors='pt')['input_ids']
                # print(tok.shape)
                tok = t.cat((bos_ten, tok[0], eos_ten))
                toks.append(tok)
            ret[f'{col}_tok'] = toks
            ret[f'{col}_length'] = [tok.shape[0] for tok in toks]
        return ret
    # ds = ds.shard(100, 1)
    kwargs = dict(tokenizer=get_tokenizer())

    ds = ds.map(map_fn, batched=True, load_from_cache_file=False, fn_kwargs=kwargs,
            num_proc=num_proc)
    ds = ds.remove_columns(('en', 'de'))
    ds = ds.sort('en_length')
    ds.set_format('torch')
    return ds

def preprocess(cache_dir, data_dir, num_proc=8, shard=None):
    """
    """
    import os
    if not os.path.exists(data_dir):
        raise RuntimeError(f'Couldn\'t find data path \'{data_dir}\'')

    print(f'Starting preprocess, data_dir={data_dir}, shard={shard}')

    ds = datasets.load_dataset('wmt14', 'de-en', cache_dir=cache_dir, split='train')
    if shard:
        ds = ds.shard(*shard)
    print(f'Tokenizing dataset')
    ds = tokenize_data(ds, num_proc)

    print(f'Computing token histo')
    de_token_histo = token_histo(ds, 'de_tok')
    t.save(de_token_histo, f'{data_dir}/de_token_histo.pt')

    print(f'Saving dataset to {data_dir}.  Columns: {ds.column_names}')
    ds.save_to_disk(data_dir)

def load_token_histo(data_path):
    histo = t.load(f'{data_path}/de_token_histo.pt')
    return histo

def get_dataset(data_path):
    return datasets.load_from_disk(data_path)

def inverse_mod(val, modulo):
    # amount needed to round up to nearest modulo
    return - (val % -modulo)

class Data(state.State):
    """
    Infinitely generates batch_size batches of indices in range(dataset_size)
    uniformly, even if dataset_size % batch_size != 0
    """
    def __init__(self):
        super().__init__()

    def make(self, params, saved_state):
        self.rng = np.random.mtrand.RandomState(params.random_seed)
        self.batch_size = params.batch_size
        self.bin_size = params.bin_size
        self.dataset_size = params.dataset_size

        if saved_state is None:
            self.offset = 0
            self.step = 0
            self.epoch = 0
        else:
            self.rng.set_state(saved_state['randstate'])
            self.offset = saved_state['offset']
            self.step = saved_state['step']
            self.epoch = saved_state['epoch']

    def get(self):
        return self

    def state(self):
        return dict(offset=self.offset, step=self.step, epoch=self.epoch,
                randstate=self.rng.get_state())

    def __iter__(self):
        while True:
            main_inds = np.arange(self.offset, self.dataset_size)
            extra = inverse_mod(main_inds.shape[0], self.batch_size) 
            wrap_inds = np.arange(extra) # wrapping around to the next epoch
            inds = np.concatenate((wrap_inds, main_inds))
            assert inds.shape[0] % self.batch_size == 0
            self.offset = extra
            inds = shuffle_inner(inds, self.bin_size, self.rng)
            inds = shuffle_outer(inds, self.batch_size, self.rng)
            for b in range(0, inds.shape[0], self.batch_size):
                yield self.epoch, self.step, inds[b:b+self.batch_size]
                self.step += 1
            self.epoch += 1

def make_batch(dataset, inds, pad_value, device):
    """
    """
    entries = dataset[inds]
    def _padded_stack(tensors, pad_value):
        max_len = max(ten.shape[0] for ten in tensors)
        st_lengths = t.tensor([ten.shape[0] for ten in tensors])
        st = t.full((len(tensors), max_len), pad_value, dtype=t.int64)
        for i, ten in enumerate(tensors):
            st[i,:ten.shape[0]] = ten 
        return st, st_lengths

    en, en_lengths = _padded_stack(entries['en_tok'], pad_value)
    de, de_lengths = _padded_stack(entries['de_tok'], pad_value)
    return en.to(device), de.to(device), en_lengths, de_lengths


if __name__ == '__main__':
    fire.Fire(preprocess)


