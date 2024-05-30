import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import torch


if __name__ == "__main__":
    # x_tf = tf.constant([[[[1, 2], [3, 40]], [[1 , -1], [2, 200]], [[3, 2], [50, 40]]]], dtype = tf.float32)
    # x_torch = torch.tensor([[[[1, 2], [3, 40]], [[1 , -1], [2, 200]], [[3, 2], [50, 40]]]], dtype=torch.float32)
    # print(x_tf.shape, x_torch.shape)

    # Torch's GroupNorm
    x = np.random.random((1, 64, 128, 128)).astype(np.float32)
    x_torch = torch.from_numpy(x)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    assert np.allclose(x_torch.detach().numpy(), x_tf.numpy(), atol=1e-5)

    num_channels = x_torch.shape[1]
    x_group_norm_torch = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)(x_torch)
    print(x_group_norm_torch)

    # Tensorflow's GroupNorm
    x_group_norm_tfa = tfa.layers.GroupNormalization(groups=1)(x_tf)
    group_norm = tf.keras.layers.GroupNormalization(groups=1)
    x_group_norm_tf = group_norm(x_tf)
    x_layer_norm_tf = tf.keras.layers.LayerNormalization(axis=1)(x_tf)
    # print(x_group_norm_tf.shape, x_group_norm_tfa.shape, x_layer_norm_tf.shape)
    print(x_group_norm_tf)

    assert np.allclose(x_group_norm_torch.detach().numpy(), x_group_norm_tf.numpy(), atol=1e-5)

    # print(">>>>>>>>>>>>> Torch >>>>>>>>>>>>>>>")
    # x = torch.tensor([[[0.9013, 0.5568, 0.8601], [0.5013, 0.6568, 0.8601]]], dtype=torch.float32)
    # indices = torch.argsort(x, dim=-1, descending=True)
    # output = torch.take_along_dim(x, indices, dim=2)
    # print(x.shape, indices.shape, output.shape)
    # print(output)

    # print(">>>>>>>>>>>>> Tensorflow >>>>>>>>>>>>>")
    # x = tf.constant([[[0.9013, 0.5568, 0.8601], [0.5013, 0.6568, 0.8601]]], dtype = tf.float32)
    # indices = tf.argsort(x, axis=-1, direction='DESCENDING')
    # print(indices[0, 0, :])
    # output = tf.gather(x, indices[0, 0, :], axis=2)
    # print(x.shape, indices.shape, output.shape)
    # print(output)
