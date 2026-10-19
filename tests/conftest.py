import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch

torch.set_num_threads(1)
