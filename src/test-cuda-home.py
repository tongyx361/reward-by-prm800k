import glob
import os
import subprocess
import sys
from typing import Optional

import torch

IS_WINDOWS = os.name == 'nt'
SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()

def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    print(f"os.environ['CUDA_HOME'] = {os.environ.get('CUDA_HOME')}")
    print(f"os.environ['CUDA_PATH'] = {os.environ.get('CUDA_PATH')}")
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                                stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                print(f"nvcc = {nvcc}")
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
                print(f"os.path.dirname(os.path.dirname(nvcc)) = {cuda_home}")
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        print(f"No CUDA runtime is found, using CUDA_HOME='{cuda_home}'",
                file=sys.stderr)
    return cuda_home

CUDA_HOME = _find_cuda_home()
print(f"CUDA_HOME = {CUDA_HOME}")