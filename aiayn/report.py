import re
import jax.numpy as jnp

"""
Functions for reporting
"""

def get_layer_values(path_pattern, values, key_name):
    """
    values:  pytree of values returned by model init function
    prefix_pattern:  complete regex to match Haiku prefix.  For example:
       tx/~/enc/~/(layer\d+)/~/att
    key_name:  name of parameter:  this will be a key name for the
       dictionary located at the prefix
    """
    pat = '^' + path_pattern + '$'
    plist = []
    for prefix, value_group in values.items():
        mat = re.match(pat, prefix)
        if not mat:
            continue
        value = value_group[key_name]
        plist.append((mat.groups(), value))
    # print(path_pattern, key_name)
    sorted_values = [par for (_, par) in sorted(plist)]
    return jnp.stack(sorted_values, axis=0)


