import jax
import os
import orbax.checkpoint
from etils import epath
from typing import Optional, Any
import tensorflow as tf

# From https://github.com/google-research/t5x/blob/main/t5x/checkpoints.py#L1889
class DatasetCheckpointHandler(orbax.checkpoint.CheckpointHandler):
    """A CheckpointHandler implementation that handles tf.data.Iterator."""

    def __init__(self, checkpoint_filename: str):
        self._checkpoint_filename = checkpoint_filename

    def save(self, directory: epath.Path, item: tf.data.Iterator):
        """Saves the given item.

          Args:
            directory: save location directory.
            item: a tf.data.Iterator to be saved.
        """
        if jax.process_count() > 1:
            directory /= f'process_{jax.process_index()}-of-{jax.process_count()}'
            directory.mkdir(parents=False, exist_ok=False)
        ckpt = tf.train.Checkpoint(ds=item)
        ckpt.write(os.fspath(directory / self._checkpoint_filename))
        # multihost_utils.sync_global_devices('DatasetCheckpointHandler:save')

    def restore(self,
            directory: epath.Path,
            item: Optional[tf.data.Iterator] = None) -> tf.data.Iterator:
        """Restores the given item.

        Args:
          directory: restore location directory.
          item: a tf.data.Iterator to be restored. Not Optional

        Returns:
          a tf.data.Iterator restored from `directory`.
        """
        if item is None:
            raise ValueError('Must provide item to restore')
        if jax.process_count() > 1:
            directory /= f'process_{jax.process_index()}-of-{jax.process_count()}'
        ckpt = tf.train.Checkpoint(ds=item)
        ckpt.read(os.fspath(directory / self._checkpoint_filename)).assert_consumed()
        return item

    def structure(self, directory: epath.Path) -> Any:
        """Unimplemented. See parent class."""
        return NotImplementedError

