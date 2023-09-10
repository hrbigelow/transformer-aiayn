import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import fire
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from tokenizers import pre_tokenizers
from tokenizers.trainers import BpeTrainer
from aiayn import data 

def get_dataset(data_dir, name, split, feature='translation', input_lang='en',
        target_lang='de'):
    builder = tfds.builder(name, data_dir=data_dir)
    builder.download_and_prepare()
    ds = builder.as_dataset(split=split, shuffle_files=False)
    ds = ds.batch(1000)
    def unpack(item):
        if feature in item:
            item = item[feature]
        return (item[input_lang], item[target_lang])
    return ds.map(unpack).unbatch(), builder.info.splits[split].num_examples

def train_tokenizer(ds, vocab_size, out_file):
    # num_elems = len(ds) * 2
    ds = ds.rebatch(1000)

    def convert(ds):
        it = ds.as_numpy_iterator()
        decode = np.vectorize(lambda x: x.decode())
        while True:
            item = next(it, None)
            if item is None:
                return
            one, two = item.values()
            yield decode(one)
            yield decode(two)

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    special_tokens = ['[UNK]', '[PAD]', '[EOS]', '[BOS]']
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(convert(ds), trainer)
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save(out_file)

def token_dataset(ds, tokenizer_file, nproc):
    """
    Create a tokenized tf.Dataset from `dataset_name` and `split`
    Use `tokenizer_file` to initialize a tokenizer
    """
    tokenizer = data.get_tokenizer(tokenizer_file)
    ds = ds.batch(1000)

    def gen(ds):
        it = ds.as_numpy_iterator() 
        unicode_decode = np.vectorize(lambda x: x.decode())
        while True:
            item = next(it, None)
            if item is None:
                return
            one, two = item
            one = tokenizer.encode_batch(unicode_decode(one))
            two = tokenizer.encode_batch(unicode_decode(two))
            yield from [(
                tf.constant(a.ids, dtype=np.uint16), 
                tf.constant(b.ids, dtype=np.uint16)) 
                for a, b in zip(one, two)]
    return gen(ds)

def write_records(data_gen, num_elem, path_template, num_shards, shards=None):
    """
    Transform all records in ds, writing them to `num_shards` separate
    `path_template` files.

    ds:  tokenized dataset
    path_template:  a relative or full path including filename stub
    num_shards:  how many separate tfrecord files to produce
    shards: iterable of shard numbers if specific shards are desired
    """
    options = tf.io.TFRecordOptions(
            compression_type=None,
            input_buffer_size=10000,
            output_buffer_size=10000)

    shards = range(num_shards)
    chunk_size = num_elem // num_shards
    begs = [chunk_size * i for i in range(num_shards)]

    chunk = -1
    for i, (t1, t2) in enumerate(data_gen):
        if chunk != i // chunk_size:
            chunk = i // chunk_size
            record_path = path_template.format(chunk)
            print(f'Writing chunk {chunk} to {record_path} of {num_shards} shards')
            file_writer = tf.io.TFRecordWriter(record_path, options)

        s1 = tf.io.serialize_tensor(t1)
        s2 = tf.io.serialize_tensor(t2)
        b1 = tf.train.BytesList(value=[s1.numpy()])
        b2 = tf.train.BytesList(value=[s2.numpy()])

        example = tf.train.Example(
            features=tf.train.Features(feature={
                'x': tf.train.Feature(bytes_list=b1),
                'y': tf.train.Feature(bytes_list=b2)
                }
            )
        )
        record_bytes = example.SerializeToString()
        file_writer.write(record_bytes)

def tokenize_dataset(download_dir, dataset_name, split, tokenizer_file, nproc,
        num_shards, out_template, input_lang, target_lang, shards=None):
    """
    Write a tfrecord dataset to out_template (must contain '{}') 
    """
    print('Preparing dataset')
    ds, num_elem = get_dataset(download_dir, dataset_name, split, 'translation', 'de', 'en')
    data_gen = token_dataset(ds, tokenizer_file, nproc)
    print('Writing tfrecords')
    write_records(data_gen, num_elem, out_template, num_shards, shards) 

if __name__ == '__main__':
    cmds=dict(tokenize=tokenize_dataset, train=train_tokenizer)
    fire.Fire(cmds)

