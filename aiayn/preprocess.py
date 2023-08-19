import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import fire
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import PreTrainedTokenizerFast

def token_dataset(download_dir, dataset_name, split, encode_fn, nproc):
    """
    """
    builder = tfds.builder(dataset_name, data_dir=download_dir)
    # the download_dir argument of download_and_prepare seems to be ignored in favor
    # of tfds.builder(data_dir=...)
    builder.download_and_prepare()
    ds = builder.as_dataset(split=split, shuffle_files=True)
    ds_info = builder.info

    def tokenize_fn(item):
        def _py_fn(one, two):
            one = encode_fn(one.numpy().decode())
            two = encode_fn(two.numpy().decode())
            return tf.constant(one, dtype=tf.uint16), tf.constant(two, dtype=tf.uint16)
        return tf.py_function(_py_fn, inp=item.values(), Tout=[tf.uint16, tf.uint16])

    ds = ds.map(tokenize_fn, num_parallel_calls=nproc, deterministic=False)
    # No need for prefetching or caching since this dataset will be
    # compiled into tfrecords
    # ds.prefetch(ds_info.splits[split].num_examples)
    # ds = ds.cache(f'{download_dir}/{dataset_name}-cache')
    # iterate once to populate cache
    return ds, ds_info

def write_records(ds, path_template, num_shards, shards=None):
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
    chunk_size = len(ds) // num_shards
    begs = [chunk_size * i for i in range(num_shards)]

    chunk = -1
    for i, (t1, t2) in enumerate(iter(ds)):
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

def main(download_dir, dataset_name, split, vocab_dir, nproc, num_shards,
        out_template, shards=None):
    """
    Write a tfrecord dataset to out_template (must contain '{}') 
    """
    from tensor2tensor import problems
    ende_problem = problems.problem('translate_ende_wmt32k')
    encoders = ende_problem.feature_encoders(vocab_dir)
    encode_fn = encoders['inputs'].encode

    # tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    print('Preparing dataset')
    ds, _ = token_dataset(download_dir, dataset_name, split, encode_fn, nproc)
    print('Writing tfrecords')
    write_records(ds, out_template, num_shards, shards) 

def encoder_stats(vocab_dir):
    from tensor2tensor import problems
    ende_problem = problems.problem('translate_ende_wmt32k')
    encoders = ende_problem.feature_encoders(vocab_dir)
    # in this case the input and target encoders are the same
    encoder = encoders['inputs']
    print(f'n_vocab: {encoder.vocab_size}')
    print(f'token 0: {encoder.decode([0])}')
    print(f'token 1: {encoder.decode([1])}')


if __name__ == '__main__':
    cmds=dict(make=main, encoder_stats=encoder_stats)
    fire.Fire(cmds)

