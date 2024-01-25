import numpy as np
import tensorflow as tf

from utils.conv_utils import deconv_output_length


class CustomConv2DTranspose(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            strides,
            padding="valid",
            output_padding=None,
            data_format="NHWC",
            dilation_rate=(1, 1),
            kernel_initializer="glorot_uniform",
            **kwargs):
        super(CustomConv2DTranspose, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.output_padding = output_padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer

        # Check if kernel_size is an integer or tuple/list of 2 integers.
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * 2
        else:
            self.kernel_size = tuple(self.kernel_size)

        # Check if strides is an integer or tuple/list of 2 integers.
        if isinstance(self.strides, int):
            self.strides = (self.strides,) * 2
        else:
            self.strides = tuple(self.strides)

        if self.data_format not in ["NHWC", "NCHW"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")

    def _get_channel_axis(self):
        if self.data_format == "NCHW":
            return 1
        else:
            return 3

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                "Inputs should have rank 4. "
                f"Received input_shape={input_shape}."
            )

        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "to `Conv2DTranspose` should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )

        # kernel_shape: (self.kernel_size[0], self.kernel_size[1], self.filters, input_dim)
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        self.built = True

    def call(self, inputs):
        # Get input shape and batch size
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0] # Tensor
        input_height, input_width, input_channels = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        if self.data_format == "channels_first":
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        # Calculate output shape
        height, width = None, None
        if inputs.shape.rank is not None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height is not None else input_shape[h_axis]
        width = width if width is not None else input_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        # deconv_output_length
        out_height = deconv_output_length(
            height,
            kernel_h,
            padding=self.padding.lower(),
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        out_width = deconv_output_length(
            width,
            kernel_w,
            padding=self.padding.lower(),
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        if self.data_format == "NCWH":
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = tf.stack(output_shape)
        output = tf.nn.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape=output_shape_tensor,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding,
            data_format=self.data_format
        )

        return output



class CustomConv2DTransposeWithConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            strides,
            padding="valid",
            output_padding=None,
            data_format="NHWC",
            dilation_rate=(1, 1),
            kernel_initializer="glorot_uniform",
            **kwargs):
        super(CustomConv2DTransposeWithConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.output_padding = output_padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer

        # Check if kernel_size is an integer or tuple/list of 2 integers.
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * 2
        else:
            self.kernel_size = tuple(self.kernel_size)

        # Check if strides is an integer or tuple/list of 2 integers.
        if isinstance(self.strides, int):
            self.strides = (self.strides,) * 2
        else:
            self.strides = tuple(self.strides)

        if self.data_format not in ["NHWC", "NCHW"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")

    def _get_channel_axis(self):
        if self.data_format == "NCHW":
            return 1
        else:
            return 3

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                "Inputs should have rank 4. "
                f"Received input_shape={input_shape}."
            )

        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "to `Conv2DTranspose` should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )

        # kernel_shape: (self.kernel_size[0], self.kernel_size[1], self.filters, input_dim)
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        self.built = True

    def call(self, inputs):
        # Get input shape and batch size
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0] # Tensor
        input_height, input_width, input_channels = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        if self.data_format == "channels_first":
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        # Calculate output shape
        height, width = None, None
        if inputs.shape.rank is not None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height is not None else input_shape[h_axis]
        width = width if width is not None else input_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        # deconv_output_length
        out_height = deconv_output_length(
            height,
            kernel_h,
            padding=self.padding.lower(),
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        out_width = deconv_output_length(
            width,
            kernel_w,
            padding=self.padding.lower(),
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        if self.data_format == "NCWH":
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = tf.stack(output_shape)
        # output = tf.nn.conv2d_transpose(
        #     inputs,
        #     self.kernel,
        #     output_shape=output_shape_tensor,
        #     strides=[1, self.strides[0], self.strides[1], 1],
        #     padding=self.padding,
        #     data_format=self.data_format
        # )

        # Upsample using nearest-neighbor (optional)
        # Adjust output shape if padding is 'same'
        # print(out_height, out_width, self.strides[0], input_height, input_width, self.kernel_size[0])
        pad_along_height = max((out_height - 1) * self.strides[0] + self.kernel_size[0] - input_height, 0)
        pad_along_width = max((out_width - 1) * self.strides[1] + self.kernel_size[1] - input_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        inputs = tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

        # rotating self.kernel clockwise by 180 degrees
        kernel = tf.reshape(self.kernel, [self.kernel_size[0], self.kernel_size[1], -1])
        kernel = tf.image.rot90(kernel, k=2)
        kernel = tf.reshape(kernel, [self.kernel_size[0], self.kernel_size[1], self.filters, -1])

        # Apply regular Conv2D operation
        output = tf.nn.conv2d(inputs, kernel,
                              strides=[1, self.strides[0], self.strides[1], 1],
                              padding=self.padding)

        return output



if __name__ == "__main__":
    x = np.array([[1, 2], [4, 5]], dtype=np.float32)
    filter = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    # x = np.array([[55, 52], [57, 50]], dtype=np.float32)
    # filter = np.array([[1, 2], [2, 1]], dtype=np.float32)

    # convert to tensor
    input_h, input_w = x.shape
    filter_h, filter_w = filter.shape
    input_dim = output_dim = 1
    input_shape = (input_h, input_w, input_dim)

    # input: [batch, height, width, channels]
    x_tf = tf.reshape(tf.convert_to_tensor(x, dtype=tf.float32), [1, input_h, input_w, input_dim])
    # filter: [height, width, output_channels, input_channels]
    filter_tf = tf.reshape(tf.convert_to_tensor(filter, dtype=tf.float32), [filter_h, filter_w, output_dim, input_dim])

    # transposed convolution using tf.keras.layers.Conv2DTranspose
    keras_model = tf.keras.models.Sequential()
    keras_model.add(
        tf.keras.layers.Conv2DTranspose(1, kernel_size=(filter_h, filter_w), strides=1, padding='valid', input_shape=input_shape)
    )

    # (weight, bias)
    weights = [filter_tf, np.asarray([0])]
    keras_model.set_weights(weights)
    keras_output = keras_model.predict(x_tf)
    keras_output = keras_output.reshape(4, 4)

    # Custom transposed convolution using `tf.nn.conv2d_transpose`
    custom_model = tf.keras.models.Sequential()
    custom_model.add(
        CustomConv2DTranspose(1, kernel_size=(filter_h, filter_w), strides=1, padding='valid', input_shape=input_shape)
    )

    # (weight,)
    weights = [filter_tf,]
    custom_model.set_weights(weights)
    custom_output = custom_model.predict(x_tf)
    custom_output = custom_output.reshape(4, 4)

    # Custom transposed convolution using `tf.nn.conv2d`
    custom_model_2 = tf.keras.models.Sequential()
    custom_model_2.add(
        CustomConv2DTransposeWithConv2D(1, kernel_size=(filter_h, filter_w), strides=1, padding='valid', input_shape=input_shape)
    )

    # (weight,)
    weights = [filter_tf,]
    custom_model_2.set_weights(weights)
    custom_output_2 = custom_model_2.predict(x_tf)
    custom_output_2 = custom_output_2.reshape(4, 4)

    print("input: \n", x)
    print("filter: \n", filter)
    print("result: \n", keras_output)
    print("custom result (tf.nn.conv2d_transpose): \n", custom_output)
    print("custom result (tf.nn.conv2d): \n", custom_output_2)
