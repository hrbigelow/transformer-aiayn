import fire
import tensorflow as tf
import jax
import os
import tempfile
import orbax.checkpoint
from orbax.checkpoint import CheckpointManagerOptions, CheckpointManager

from aiayn import data, utils


def main(data_dir, dataset_name, token_info_file):
    rng = jax.random.PRNGKey(42)

    data.set_config(data_dir=data_dir)
    token_info = data.load_token_info(token_info_file)
    ds = tf.data.Dataset.load(os.path.join(data_dir, dataset_name))
    pad_ds = data.pad_dataset(ds, token_info, 1000, True, 100)
    path = '/home/henry/ai/data/abcde'

    it = iter(pad_ds)
    it2 = iter(pad_ds)

    for _ in range(100):
        _ = next(it)


    checkpointers = { 'dataset': utils.DatasetCheckpointHandler('dataset') }
    mngr = CheckpointManager(path, checkpointers)
    mngr.save(100, {'dataset': it }, save_kwargs={ 'dataset': {} })

    print('next it')
    print(next(it))

    restored = mngr.restore(100, { 'dataset': it2 })

    print('next it2')
    print(next(it2))

def reload():
    path = '/home/henry/ai/data/abcde'
    checkpointers = { 'dataset': utils.DatasetCheckpointHandler('dataset') }
    mngr = CheckpointManager(path, checkpointers)

    restored = mngr.restore(100)
    print(restored)
        
if __name__ == '__main__':
    fire.Fire(dict(main=main, reload=reload))
