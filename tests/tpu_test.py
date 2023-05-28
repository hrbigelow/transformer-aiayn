import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def report(res):
    print(f'{res.shape=}')

def gather_and_report(ten):
    result = xm.all_gather(ten, dim=0)
    print(f'{result=}')

def _mp_fn(index):
    # print(xm.xrt_world_size())
    device = xm.xla_device()
    ordinal_tensor = torch.randn((5,3), device=device)
    # ordinal_tensor = torch.tensor([index], dtype=torch.float).to(device)

    result = xm.all_gather(ordinal_tensor, dim=0)
    # print(f'{cpu_result=}')

    # this hangs
    # xm.do_on_ordinals(gather_and_report, (ordinal_tensor,))

    xm.do_on_ordinals(report, (result,))

if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=())

