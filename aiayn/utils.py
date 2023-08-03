import jax
import os
import optax
import orbax.checkpoint
from orbax.checkpoint import type_handlers, SaveArgs
from etils import epath
from typing import Optional, Any, NamedTuple, List
import tensorflow as tf

class NamedTupleHandler(type_handlers.TypeHandler):
  def __init__(self, cls):
    self.cls = cls

  async def serialize(
      self,
      value: NamedTuple,
      info: type_handlers.ParamInfo,
      args: Optional[SaveArgs] = None) -> List[orbax.checkpoint.future.Future]:
    # A more sophisticated implementation would make this write asynchronous.
    (info.path / 'data.txt').write_text(value._asdict())
    print('got here in serialize')
    return []

  async def deserialize(
      self,
      info: type_handlers.ParamInfo,
      args: Optional[orbax.checkpoint.RestoreArgs] = None):
    entries = (info.path / 'data.txt').read_text()
    return self.cls(**entries)


def register_handlers():
    type_handlers.register_type_handler(optax.ScaleByAdamState,
            NamedTupleHandler(optax.ScaleByAdamState), override=True)
    type_handlers.register_type_handler(optax.ScaleByScheduleState,
            NamedTupleHandler(optax.ScaleByScheduleState), override=True)


def construct_restore_args(target, sharding_tree, set_global_shape=True):
    """
    Shim from orbax.checkpoint.checkpoint_utils since that relies on
    jax.sharding.Mesh, which is not present in jax 0.3.25
    """

    def _restore_args(value: Any, sharding: jax.sharding.Sharding):
        restore_type = type(value)
        dtype = None
        if hasattr(value, 'dtype'):
            dtype = value.dtype
        if isinstance(value, jax.Array):
            return type_handlers.ArrayRestoreArgs(
                    restore_type=restore_type,
                    # sharding=sharding,
                    global_shape=value.shape if set_global_shape else None,
                    dtype=value.dtype
          )
        else:
            return type_handlers.RestoreArgs(restore_type=restore_type, dtype=dtype)

    return jax.tree_util.tree_map(_restore_args, target, sharding_tree)

