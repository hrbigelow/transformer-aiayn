import sys
# import torch.multiprocessing as mp
import multiprocessing as mp
import queue
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend
import signal
import time


def test_handler(signum, frame):
    print(f'in test_handler')

def mp_func(index):
    signal.signal(signal.SIGINT, test_handler)

    # print(f'in mp_func: {id(parent_queue)=}')
    step = 0
    while True:
        time.sleep(1)
        print(f'on {step=}')
        # timeout is zero
        step += 1

def main():
    # parent_queue = mp.Queue()
    # child_queue = mp.Queue()
    # print(f'in main: {id(parent_queue)=}')

    def shutdown_handler(signum, frame):
        parent_queue.put('cleanup')
        try:
            child_msg = child_queue.get(block=True, timeout=3)
            print(f'child responded with {child_msg=}')
            sys.exit(0)
        except queue.Empty:
            print(f'child did not respond')

    # signal.signal(signal.SIGINT, shutdown_handler)
    # signal.signal(signal.SIGINT, test_handler)
    
    xmp.spawn(mp_func, args=(), nprocs=1)

if __name__ == '__main__':
    main()




