import torch
import numpy as np


if __name__ == "__main__":
    # Intializing a Tensor
    ## 1. Directly from data
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)

    ## 2. From a Numpy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    ## 3. From another tensor
    x_ones = torch.ones_like(x_data) # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")

    ## 4. Create a Tensor with empty, random, or constant values
    shape = (2,3,)
    empty_tensor = torch.empty(shape)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Empty Tensor: \n {empty_tensor} \n")
    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

    # Attributes of a Tensor
    tensor = torch.rand(3,4)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    tensor = torch.tensor([[1, 1], [2, 2]])

    # Arithmetic operations (算術運算)
    ## 1. matrix multiplication
    out1 = tensor @ tensor
    print(out1)

    ## 2. element-wise product
    out2 = tensor * tensor
    print(out2)

    # Single-element tensors
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))

    # In-place operations
    print(f"{tensor} \n")
    tensor.add_(5)
    print(tensor)

    # Bridge with NumPy
    ## 1. Tensor to NumPy array
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")

    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")
