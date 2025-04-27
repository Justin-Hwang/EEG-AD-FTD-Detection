import gc
import torch

gc.collect()
torch.mps.empty_cache()  # MPS 환경일 경우 (Mac M1/M2)
