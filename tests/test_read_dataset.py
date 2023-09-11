import fire
import tensorflow as tf
from tokenizers import Tokenizer, decoders

from aiayn import data, pack

def print_data(dataset_glob, tokenizer_file, swap_source_target, num_elem=-1):
    tokenizer = Tokenizer.from_str(tf.io.gfile.GFile(tokenizer_file).read())
    # tokenizer.enable_padding(pad_id=special_toks.pad_id, pad_token='[PAD]')
    tokenizer.decoder = decoders.ByteLevel()
    ds = data.load_tfrecord_dataset(dataset_glob, swap_source_target)
    if num_elem >= 0:
        ds = ds.take(num_elem)
    it = ds.as_numpy_iterator()
    for item in it:
        input = tokenizer.decode(item['inputs'])
        target = tokenizer.decode(item['targets'])
        print(input)
        print(target)
        print()
        # print('\n'.join(inputs))

def hash_data(ds_iter):
    result = {} 
    for item in ds_iter:
        t = (tuple(item['inputs']), tuple(item['targets']))
        result[hash(t)] = t
    return result

def validate_packing(dataset_glob, swap_source_target, num_elem):
    feature_lengths = { 'inputs': 5000, 'targets': 5000 }

    ds = data.load_tfrecord_dataset(dataset_glob, swap_source_target)
    ds = ds.take(num_elem)
    it = ds.as_numpy_iterator()
    orig_result = hash_data(it)

    pad_value = -2
    num_tries = 10
    pack_batch_size = 1000
    pack_ds = pack.pack_dataset(ds, feature_lengths, pack_batch_size, num_tries, pad_value)
    unpack_ds = pack.unpack_dataset(pack_ds, num_tries, pad_value)
    un_it = unpack_ds.as_numpy_iterator()

    rt_result = hash_data(un_it)
    for orig_k, orig_v in orig_result.items():
        if orig_k not in rt_result:
            print(f'Missing from pack: {len(orig_v[0])}, {len(orig_v[1])}')
        elif rt_result[orig_k] != orig_v:
            print(f'Differs: {orig_v}\n{rt_result[orig_k]}')

    for rt_k, rt_v in rt_result.items():
        if rt_k not in orig_result:
            print(f'Missing from orig: {rt_v}')
        elif orig_result[rt_k] != rt_v:
            print(f'Differs: {rt_v}\n{orig_result[rt_k]}')

            
    # diff = orig_result.difference(rt_result)
    # ixn = orig_result.intersection(rt_result)
    # print(f'{len(ixn)} of {len(orig_result)} matched')
    # print(f'{len(orig_result)=} {len(rt_result)=}')

if __name__ == '__main__':
    cmds = dict(print_data=print_data, validate_packing=validate_packing)
    fire.Fire(cmds)

