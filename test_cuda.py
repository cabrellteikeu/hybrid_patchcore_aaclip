import torch
import torchvision
import PIL
import numpy

print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"PIL/Pillow: {PIL.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")