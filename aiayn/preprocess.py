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

def train_tokenizer(download_dir, dataset_name, vocab_size, out_file):
    """
    Train a BPE tokenizer on `dataset_name` with `vocab_size` tokens.

    Train a BPE tokenizer on the sentence pairs from the train split of
    `dataset_name`.  Train it to approximately `vocab_size` tokens, and save the
    trained tokenizer to `out_file` which should have '.json'.

    :param download_dir: Directory for caching downloaded dataset 
    :param dataset_name: Name of dataset (as listed by tfds.list_builders())
    :param vocab_size: Desired vocabulary size
    :param out_file: '.json' file to save trained tokenizer

    """
    builder = tfds.builder(dataset_name, data_dir=download_dir)
    builder.download_and_prepare()
    ds = builder.as_dataset(split='train', shuffle_files=False)
    num_examples = builder.info.splits['train'].num_examples
    ds = ds.batch(1000)

    def convert(ds):
        it = ds.as_numpy_iterator()
        decode = np.vectorize(lambda x: x.decode())
        while True:
            item = next(it, None)
            if item is None:
                return
            if 'translation' in item:
                item = item['translation']
            one, two = item.values()
            yield decode(one)
            yield decode(two)

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    special_tokens = ['[UNK]', '[PAD]', '[EOS]', '[BOS]']
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(convert(ds), trainer, length=num_examples*2)
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save(out_file)
    print(f'Saved tokenizer to {out_file}')

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
    shard_size = num_elem // num_shards
    begs = [shard_size * i for i in range(num_shards)]
    file_writer = None

    prev_shard = None
    for i, (t1, t2) in enumerate(data_gen):
        shard = i // shard_size
        if shard != prev_shard:
            if file_writer is not None:
                file_writer.flush()
                file_writer.close()
            record_path = path_template.format(shard)
            print(f'Writing shard {shard} to {record_path} of {num_shards} shards')
            file_writer = tf.io.TFRecordWriter(record_path, options)
            prev_shard = shard

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
    file_writer.flush()
    file_writer.close()

def tokenize_dataset(download_dir, dataset_name, split, tokenizer_file, nproc,
        num_shards, out_template, input_lang, target_lang):
    """
    Write a tfrecord dataset to out_template (must contain '{}') 
    """
    print('Preparing dataset')
    ds, num_elem = get_dataset(download_dir, dataset_name, split, 'translation', 'de', 'en')
    data_gen = token_dataset(ds, tokenizer_file, nproc)
    print('Writing tfrecords')
    write_records(data_gen, num_elem, out_template, num_shards) 

if __name__ == '__main__':
    cmds=dict(tokenize_dataset=tokenize_dataset, train_tokenizer=train_tokenizer)
    fire.Fire(cmds)

