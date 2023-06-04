"""
Synopsis:

    spawn child processes using xmp.spawn
    test what happens when the child:
      - raises an Exception
      - gets killed with SIGKILL
"""


import sys, os, signal
import time
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(rank, mode):
    print(f'In _mp_fn {rank=}, pid={os.getpid()}')
    time.sleep(3)
    if mode == 'sigkill':
        os.kill(os.getpid(), signal.SIGKILL.value)
    elif mode == 'raise':
        raise RuntimeError(f'_mp_fn {rank=} raised error')

def main():
    mode, use_pjrt = sys.argv[1:]
    if use_pjrt == 'yes':
        os.environ['PJRT_DEVICE'] = 'GPU'
        os.environ['GPU_NUM_DEVICES'] = '1'
    else:
        del os.environ['PJRT_DEVICE']
        del os.environ['GPU_NUM_DEVICES']

    print(f'Starting main(), {use_pjrt=}, pid={os.getpid()}')
    xmp.spawn(_mp_fn, args=(mode,))

if __name__ == '__main__':
    main()




