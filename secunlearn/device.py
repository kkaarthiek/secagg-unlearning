"""Which processor to use, and the settings that make a redone round reproduce the original.

CUDA when there is one, CPU otherwise. Apple's MPS is deliberately never used: redoing a
round must reproduce the original to the last bit, and MPS makes no promise about that.
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One CPU thread: the thread count changes the order numbers are added in, and that changes
# the last bit of the answer.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
