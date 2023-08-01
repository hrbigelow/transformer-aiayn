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

