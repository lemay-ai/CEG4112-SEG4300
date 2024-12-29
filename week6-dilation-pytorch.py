!pip install icecream torch
import torch
from icecream import ic

# Create a 10x10 tensor with increasing values
tensor = torch.arange(100).reshape(10, 10)

ic(tensor)

# Define kernel size and dilation
kernel_size = (3, 3)
for dilation in range(1, 4):
  # Calculate the effective kernel size with dilation
  effective_kernel_size = ((kernel_size[0] - 1) * dilation + 1, (kernel_size[1] - 1) * dilation + 1)

  # Calculate padding to maintain output size
  padding = ((effective_kernel_size[0] - 1) // 2, (effective_kernel_size[1] - 1) // 2)

  # Iterate and print values within the kernel with dilation
  strategy = f"Values accessed by a {kernel_size} kernel with dilation {dilation}:"
  ic(strategy)
  for row in range(tensor.shape[0] - (kernel_size[0] - 1) * dilation):
      for col in range(tensor.shape[1] - (kernel_size[1] - 1) * dilation):
          # Calculate the indices for the dilated kernel
          rows = [row + i * dilation for i in range(kernel_size[0])]
          cols = [col + j * dilation for j in range(kernel_size[1])]

          # Extract values using advanced indexing
          kernel_view = tensor[torch.tensor(rows)[:,None], torch.tensor(cols)]
          ic(kernel_view)
          break
      break