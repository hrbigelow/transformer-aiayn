import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def report(ten):
    combined = xm.all_reduce(xm.REDUCE_SUM, ten)
    xm.mark_step()

    if xm.is_master_ordinal():
        print(f'master: got here with {combined.shape=}', flush=True)
        cpu_result = combined.cpu()
        print(f'{cpu_result=}', flush=True)
    else:
        print(f'subbordinate: got here with {combined.shape=}', flush=True)

def _mp_fn(index):
    dev = xm.xla_device()
    ten = torch.randn((10, 30), device=dev)

    xm.add_step_closure(report, args=(ten,)) 
    xm.mark_step()


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=()) 
