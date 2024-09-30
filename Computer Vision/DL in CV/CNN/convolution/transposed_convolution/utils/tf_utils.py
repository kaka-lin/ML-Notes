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


def convert_data_format(data_format, ndim):
    if data_format == "channels_last":
        if ndim == 3:
            return "NWC"
        elif ndim == 4:
            return "NHWC"
        elif ndim == 5:
            return "NDHWC"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    elif data_format == "channels_first":
        if ndim == 3:
            return "NCW"
        elif ndim == 4:
            return "NCHW"
        elif ndim == 5:
            return "NCDHW"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    else:
        raise ValueError(
            f"Invalid data_format: {data_format}. "
            'Expected values are ["channels_first", "channels_last"]'
        )


def normalize_tuple(value, n, name, allow_zero=False):
    """Transforms non-negative/positive integer/integers into an integer tuple.

    Args:
      value: The value to validate and convert. Could an int, or any iterable of
        ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
      allow_zero: A ValueError will be raised if zero is received
        and this param is False. Defaults to `False`.

    Returns:
      A tuple of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof or a
      negative value is
        passed.
    """
    error_msg = (
        f"The `{name}` argument must be a tuple of {n} "
        f"integers. Received: {value}"
    )

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (
                    f"including element {single_value} of "
                    f"type {type(single_value)}"
                )
                raise ValueError(error_msg)

    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = ">= 0"
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = "> 0"

    if unqualified_values:
        error_msg += (
            f" including {unqualified_values}"
            f" that does not satisfy the requirement `{req_msg}`."
        )
        raise ValueError(error_msg)

    return value_tuple
