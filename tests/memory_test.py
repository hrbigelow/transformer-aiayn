import torch
import psutil

"""
Test the difference between:

a = a.to(dev)
a.to(dev)

on memory usage.

"""

def print_mem(preamble):
    rss_gb = psutil.Process().memory_info().rss / 1024 ** 3
    print(f'{preamble:50}{rss_gb:.2f} GB CPU')


# 1 GB tensor

def main():
    print_mem(f'Starting')
    a = torch.randn((1024, 1024, 256), dtype=torch.float32)
    print_mem(f'Allocated 1GB tensor on CPU')
    a.to(torch.device('cuda'))
    print_mem(f'Moved using a.to()')
    del a
    print_mem(f'deleted tensor via \'del a\'')
    b = torch.randn((1024, 1024, 256), dtype=torch.float32)
    print_mem(f'Created 1GB tensor on CPU')
    b = b.to(torch.device('cuda'))
    print_mem(f'Moved using b = b.to()')

if __name__ == '__main__':
    main()






