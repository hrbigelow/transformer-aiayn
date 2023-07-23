
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

