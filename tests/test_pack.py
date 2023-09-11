import tensorflow as tf
import fire
from aiayn import data, pack

def main(glob, pack_batch_size, num_tries, max_source_len, max_target_len, pad_value):
    feature_lengths = { 'inputs': max_source_len, 'targets': max_target_len }
    ds = data.load_tfrecord_dataset(glob, False)
    total_items = sum(1 for _ in ds.as_numpy_iterator())
    pack_ds = pack.pack_dataset(ds, feature_lengths, pack_batch_size, num_tries, pad_value)
    pit = pack_ds.as_numpy_iterator()
    
    # pit = pack_ds.batch(10).take(15).as_numpy_iterator()
    # for item in pit:
        # shapes = tf.nest.map_structure(lambda x: x.shape, item) 
        # print(shapes)
    # return
    # pack_ds = pack_ds.take(85)
    unpack_ds = pack.unpack_dataset(pack_ds, num_tries, pad_value)
    # uit = unpack_ds.as_numpy_iterator()
    # for item in uit:
        # shapes = tf.nest.map_structure(lambda x: x.shape, item) 
        # print(shapes)

    # print(next(uit))
    # print(next(uit))
    total_unpacked_items = sum(1 for _ in unpack_ds.as_numpy_iterator())
    print(f'{total_items=}, {total_unpacked_items=}')


if __name__ == '__main__':
    fire.Fire(main)

