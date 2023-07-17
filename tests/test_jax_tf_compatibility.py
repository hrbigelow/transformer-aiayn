import subprocess
import os

def get_loaded_libs():
    cmd = ['lsof', '-p', str(os.getpid())]
    libs = subprocess.run(cmd, capture_output=True)
    lines = libs.stdout.decode().split('\n')

"""
Summary:

    The call to tf.data.Dataset.from_tensor_slices somehow alters the environment
    so that the call to jax.random.PRNGKey() fails.


"""
# RESULTS:
# IT, IJ, RT, RJ - FAILED_PRECONDITION
# IT, IJ, RJ, RT - works 
# IT, RT, IJ, RJ - FAILED_PRECONDITION 
# IJ, IT, RT, RJ - FAILED_PRECONDITION 
# IJ, RJ, IT, RT - works 
# IJ, IT, RJ     - works 
# IJ, IT, RT     - works 

import tensorflow as tf # IT
# ds = tf.data.Dataset.from_tensor_slices([1,2,3])  #RT
import jax              # IJ
# del ds
del tf
rng_key = jax.random.PRNGKey(42)                  #RJ

"""
libs2 = subprocess.run(cmd, capture_output=True)
libs3 = subprocess.run(cmd, capture_output=True)
libs4 = subprocess.run(cmd, capture_output=True)
libs5 = subprocess.run(cmd, capture_output=True)
"""

