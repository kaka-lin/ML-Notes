import tensorflow as tf


def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Copy of the function of keras-team/keras because it's not in the public API
    So we can't use the function in keras-team/keras to test tf.keras

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def deconv_output_length(
    input_length,
    filter_size,
    padding,
    output_padding=None,
    stride=0,
    dilation=1,
):
    """Determines output length of a transposed convolution given input length.

    Args:
        input_length: Integer.
        filter_size: Integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        output_padding: Integer, amount of padding along the output dimension.
          Can be set to `None` in which case the output length is inferred.
        stride: Integer.
        dilation: Integer.

    Returns:
        The output length (integer).
    """
    assert padding in {"same", "valid", "full"}
    if input_length is None:
        return None

    # Get the dilated kernel size
    filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == "valid":
            length = input_length * stride + max(filter_size - stride, 0)
        elif padding == "full":
            length = input_length * stride - (stride + filter_size - 2)
        elif padding == "same":
            length = input_length * stride

    else:
        if padding == "same":
            pad = filter_size // 2
        elif padding == "valid":
            pad = 0
        elif padding == "full":
            pad = filter_size - 1

        length = (
            (input_length - 1) * stride + filter_size - 2 * pad + output_padding
        )
    return length
