import re
import jax
import os
import optax
import orbax.checkpoint
from orbax.checkpoint import type_handlers, SaveArgs
from etils import epath
from typing import Optional, Any, NamedTuple, List
import tensorflow as tf

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


def shape_check(index_expr, **tensors):
    """
    Check that the shapes of `tensors` are consistent with `index_expr`

    index_expr: string.  A comma-separated list of index signatures.  Each signature
    is a sequence of single-letter indexes as would be used in einsum.
    The signature must match [a-zA-Z]*
    """
    sigs = re.split(' *, *', index_expr)
    if len(sigs) != len(tensors):
        raise RuntimeError(f'Got {len(sigs)} signatures but {len(tensors)} tensors')

    z = zip(sigs, tensors.items())
    sig_shape = { arg: (sig, val.shape) for sig, (arg, val) in z }

    usage_map = {} # index => { size => [name, ...] }
    for arg, (sig, shape) in sig_shape.items(): 
        if not re.match('^[a-zA-Z]*$', sig):
            raise RuntimeError(f'Signature `{sig}` does not match `^[a-zA-Z]*$`')
        if len(sig) != len(shape):
            raise RuntimeError(
                    f'Signature `{sig}` has {len(sig)} indices but tensor '
                    f'`{arg}` with shape {shape} has {len(shape)} dimensions')
        for index, dimension in zip(sig, shape):
            entry = usage_map.setdefault(index, {})
            args = entry.setdefault(dimension, [])
            args.append(arg)
    # print(sig_shape['arg1'])

    def annotated_shape(arg):
        ind_dim = zip(*sig_shape[arg])
        # print(list(ind_dim))
        return ', '.join(f'{index}={dimension}' for index, dimension in ind_dim)

    # Validate
    for index, usage in usage_map.items():
        if len(usage) > 1:
            all_args = set(arg for args in usage.values() for arg in args)
            ordered_args = [arg for arg in tensors.keys() if arg in all_args]
            raise RuntimeError(
                f'Index `{index}` shape mismatch: ' +
                ', '.join(f'{arg}:[{annotated_shape(arg)}]' for arg in ordered_args)
                )
    return

