!pip install icecream torch tensorflow
import tensorflow as tf
import torch
import numpy as np
from icecream import  ic
data = np.array([[[1, 2, 3], [4, 5, 6]]])
ic(data)

# Example with tf.keras.layers.Flatten
# Define a sample 2D array
dimensions = "(batch_size, height, width)"
tf_tensor = tf.constant(data, dtype=tf.float32)
ic(tf_tensor.shape,dimensions)

# Create a Flatten layer
flatten_layer = tf.keras.layers.Flatten()
# Flatten the tensor
flattened_tf_tensor = flatten_layer(tf_tensor)
ic(flattened_tf_tensor)

# Example with PyTorch

torch_tensor = torch.tensor(data, dtype=torch.float32)
ic(torch_tensor.shape,dimensions)
# Flatten the tensor using view()
flattened_torch_tensor = torch_tensor.view(-1)
ic(flattened_torch_tensor)
# Alternatively, use flatten()
flattened_torch_tensor_2 = torch.flatten(torch_tensor)
ic(flattened_torch_tensor_2)
np.array_equal(flattened_torch_tensor_2,flattened_torch_tensor)