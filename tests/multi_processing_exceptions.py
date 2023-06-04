import time
import sys
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(rank):
    if rank % 2 == 0:
        raise Exception
    while True:
        time.sleep(1)
        print(f'In worker {rank=}')

if __name__ == '__main__':
    infra = sys.argv[1]
    if infra in ('tpu_vm', 'gpu'):
        xmp.spawn(_mp_fn, args=())
    elif infra == 'tpu_colab':
        xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')
    else:
        pass




