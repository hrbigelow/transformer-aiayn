import fire
from pprint import pprint
from aiayn import pack, data, hparams
import tensorflow as tf
from collections import defaultdict

def yield_test(ds, reps):
    it = ds.as_numpy_iterator()
    for i in range(reps):
        item = next(it)
        print(i)

def get_counts(ds, reps, shape):
    counts = { 'inputs': tf.zeros(shape, tf.int32), 
            'targets': tf.zeros(shape, tf.int32) }
    # count = { 'inputs': tf.zeros(, 'targets' 0 }
    def sum_fn(counts, item):
        inputs = tf.add(counts['inputs'], item['inputs']['counts'])
        targets = tf.add(counts['targets'], item['targets']['counts'])
        return dict(inputs=inputs, targets=targets)
    return ds.take(reps).reduce(counts, sum_fn)

def main(batch_size, num_tries, inputs_len, targets_len):
    dataset_glob = '/home/henry/ai/data/de-en.tfrec/*'
    feature_lengths = { 'inputs': inputs_len, 'targets': targets_len}
    toks_ds = data.load_tfrecord_dataset(dataset_glob, True)
    pack_ds = pack.pack_dataset(toks_ds, feature_lengths, batch_size, num_tries, -1)
    it = pack_ds.as_numpy_iterator()

    # This causes all sorts of problems:
    # pprint(tf.nest.map_structure(tf.shape, next(it)))

    # But, this works fine
    item = next(it)
    pprint(item)
    print('target seqs: ')
    tf.print(item['targets']['seqs'][0], summarize=-1)
    pprint(tf.nest.map_structure(tf.shape, item))
    # it = size_ds.as_numpy_iterator()
    # pprint(pack_ds.element_spec)
    # yield_test(pack_ds, 100)
    reps = 50
    # total_counts = get_counts(pack_ds, 50, (batch_size, num_tries))
    # max_counts = { k: v * reps * batch_size for k, v in feature_lengths.items() }
    # pprint(total_counts)
    # totals = tf.nest.map_structure(tf.reduce_sum, total_counts)
    # pprint(totals)
    # pprint(max_counts)
    # fracs = tf.nest.map_structure(tf.truediv, totals, max_counts)
    # pprint(fracs)


if __name__ == '__main__':
    fire.Fire(main)

