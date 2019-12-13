"""
Model definition
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def get_n_channels(i, start_channels=64, inverted=False):
    """
    Get number of channels to use at specific level
    
    Args:
        i (int): current level
        start_channels (int, optional): number of channels of first level, 
            default 64
        inverted (bool, optional): whether to use inverted architecture, 
            default False
    
    Returns:
        int: number of channels to use at level i
    """
    start_index = int(np.log(start_channels) / np.log(2))
    mul = -1 if inverted else 1
    return int(2 ** (mul*i + start_index))


def init_stddev(n, init='he'):
    """
    Standard deviation to use based on initialization scheme
    
    Args:
        n (init): number of input neurons
        init (str, optional): initialization to use, one of 'he' or 'xavier',
            default 'he'
    
    Returns:
        float: standard deviation
    
    Raises:
        ValueError: invalid initialization scheme
    """
    if init == 'he':
        return np.sqrt(2/n)
    elif init == 'xavier':
        return np.sqrt(1/n)
    else:
        raise ValueError("Invalid Initialization")


def init(input, conv_kernel=[3, 3, 3], init='he'):
    """
    get initializer for particular input, filter size and initialization scheme
    
    Args:
        input (tf.Tensor): input tensor
        conv_kernel (list, optional): kernel size
        init (str, optional): initialization scheme to use, default 'he'
    
    Returns:
        tf.truncated_normal_initializer: initializer
    """
    in_channels = input.shape[-1]
    stddev = init_stddev(int(in_channels * np.prod(conv_kernel)), init)
    return tf.truncated_normal_initializer(stddev=stddev)


def conv_sequence(x, i, is_training=False,
                  n_convs=2, start_channels=64,
                  conv_kernel=[3, 3, 3], inverted=False,
                  dropout_rate=0):
    """
    build sequence of convolutions for a level of the contracting 
    path of the network
    
    Args:
        x (tf.Tensor): input tensor
        i (int): current level
        is_training (bool, optional): whether is training, default False
        n_convs (int, optional): number of convoltions to perform, default 2
        start_channels (int, optional): channels of first level of the network, 
            default 64
        conv_kernel (list, optional): size of convolution kernel, default 3*3
        inverted (bool, optional): whether to use inverted architecture,
            default False
        dropout_rate (float, optional): rate of gaussian dropout, default 0
    
    Returns:
        tf.Tensor: output of cantracting path of level
    """
    n_channels = get_n_channels(
        i, start_channels=start_channels, inverted=inverted)

    for c in range(n_convs):
        x = slim.conv2d(x, n_channels, conv_kernel,
                        weights_initializer=init(x, conv_kernel=conv_kernel),
                        scope='conv_down_%d_%d' % (i, c))
        x = gaussian_dropout(x, dropout_rate, is_training=is_training)

    return x


def upconv_sequence(x, i, convs, is_training=False,
                    n_convs=2, start_channels=64,
                    conv_kernel=[3, 3, 3], inverted=False,
                    dropout_rate=0,
                    upscale_method="convolution3d_transpose"):
    """
    build sequence of convolutions for a level of the expanding path of 
    the network
    
    Args:
        x (tf.Tensor): input tensor
        i (int): current level
        is_training (bool, optional): whether is training, default False
        n_convs (int, optional): number of convoltions to perform, default 2
        start_channels (int, optional): channels of first level of the network,
            default 64
        conv_kernel (list, optional): size of convolution kernel, default 3*3
        inverted (bool, optional): whether to use inverted architecture, 
            default False
        dropout_rate (float, optional): rate of gaussian dropout, default 0
        upscale_method (str, optional): upscaling method to use, 
            one of 'convolution2d_transpose' or 'resize', 
            default 'convolution2d_transpose'
    
    Returns:
        tf.Tensor: output of cantracting path of level
    
    Raises:
        ValueError: invalid upscaling method
    """
    n_channels = get_n_channels(
        i - 1, start_channels=start_channels, inverted=inverted)

    if upscale_method == "convolution2d_transpose":
        # upsample with transpose convolution
        x = slim.convolution2d_transpose(
            x, n_channels, [2, 2], stride=2,
            weights_initializer=init(x, conv_kernel=[2, 2]),
            padding='VALID',
            scope="conv_t_%d" % i
        )

    elif upscale_method == "convolution3d_transpose":
        # upsample with transpose convolution
        x = slim.convolution3d_transpose(
            x, n_channels, [2, 2, 2], stride=2,
            weights_initializer=init(x, conv_kernel=[2, 2, 2]),
            padding='VALID',
            scope="conv_t_%d" % i
        )

    elif upscale_method == "resize":
        # upsample with nearest neighbor followed by convolution
        x = tf.image.resize_images(
            x,
            np.array(x.shape[1:3]) * 2,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        x = slim.conv2d(x, n_channels, [1, 1],
                        weights_initializer=init(
                            x, conv_kernel=[1, 1]),
                        scope="resize_conv_%d" % i)

    elif upscale_method == "resize3d":
        # upsample with nearest neighbor followed by convolution
        x = resize3d(
            x,
            factor=2,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        x = slim.conv2d(x, n_channels, [1, 1, 1],
                        weights_initializer=init(
                            x, conv_kernel=[1, 1, 1]),
                        scope="resize_conv_%d" % i)

    else:
        raise ValueError("Invalid Upscale Method")

    # concatenate activation map from contracting path of same level
    x = tf.concat([convs[i-1], x], axis=len(x.shape) - 1, name="concat_%d" % i)

    # sequence of convolutions of this level
    for c in range(n_convs):
        x = slim.conv2d(x, n_channels, conv_kernel,
                        weights_initializer=init(x, conv_kernel=conv_kernel),
                        scope='conv_up_%d_%d' % (i, c))
        x = gaussian_dropout(x, dropout_rate, is_training=is_training)
    return x


def gaussian_dropout(input, stddev, is_training=False):
    """
    apply gaussian dropout
    
    Args:
        input (tf.Tensor): input
        stddev (float): standard deviation of gaussian
        is_training (bool, optional): whether currently training, default False
    
    Returns:
        tf.Tensor: output
    
    Raises:
        ValueError: invalid value for standard deviation
    """
    if stddev < 0:
        raise ValueError("Invalid stddev")
    if stddev == 1:
        return input
    noise = tf.random_normal(
        shape=tf.shape(input),
        mean=1.,
        stddev=stddev,
        dtype=tf.float32
    )
    if isinstance(is_training, bool):
        if is_training:
            return input * noise
        else:
            return input
    else:
        return tf.cond(is_training, lambda: input * noise, lambda: input)


def model(x, is_training=False, n_classes=5, start_channels=64,
          n_levels=5, n_convs=2, use_elu=False, reg=0.,
          conv_kernel=[3, 3, 3], inverted=False, all_conv=False,
          use_batch_norm=False, dropout_rate=0.,
          upscale_method="convolution3d_transpose"):
    """
    apply the u-net model
    
    Args:
        x (tf.Tensor): input tensor, 
            of shape [batch_num, height, width, n_channels]
        is_training (bool, optional): whether is training, default False
        n_classes (int, optional): number of classes of output, default 5
        start_channels (int, optional): channels of first level of 
            the network (64 in original paper), default 64
        n_levels (int, optional): number of levels in architecture 
            (5 in original paper), default 5
        n_convs (int, optional): number of convoltions to perform 
            (2 in original paper), default 2
        use_elu (bool, optional): if true, use ELU as activation function,
            else use RELU, default false
        reg (float, optional): regularization value, default 0
        conv_kernel (list, optional): size of convolution kernel, default 3*3
        inverted (bool, optional): if true, use inverted architecture, 
            default False
        all_conv (bool, optional): if true, replace max pool at downsampling 
            steps with convolutions, default False
        use_batch_norm (bool, optional): if true, use bathc normalization, 
            default False
        dropout_rate (float, optional): rate of gaussian dropout, default 0
        upscale_method (str, optional): upscaling method to use, one of 
            'convolution2d_transpose' or 'resize', 
            default 'convolution2d_transpose'

    Returns:
        tf.Tensor: output logits, of shape [batch_num, height, width, n_classes]
    """

    if use_batch_norm:
        normalizer_fn = slim.batch_norm
    else:
        normalizer_fn = None


    if use_elu:
        activation_fn = tf.nn.elu
    else:
        activation_fn = tf.nn.relu


    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose, slim.convolution3d_transpose],
                        padding='SAME',
                        normalizer_fn=normalizer_fn,
                        activation_fn=activation_fn,
                        weights_regularizer=slim.l2_regularizer(float(reg))):

        convs = []

        # build contracting path
        for i in range(n_levels):
            x = conv_sequence(x, i,
                              is_training=is_training,
                              n_convs=n_convs,
                              start_channels=start_channels,
                              conv_kernel=conv_kernel,
                              inverted=inverted,
                              dropout_rate=dropout_rate)

            # don't downsample last level
            if i != n_levels - 1:
                convs.append(x)
                if all_conv:
                    # downsample with convolution
                    down_kernel = [2 for dim in conv_kernel]
                    x = slim.conv2d(
                        x, x.shape[-1], down_kernel,
                        stride=2,
                        activation_fn=None,
                        weights_initializer=init(
                            x, conv_kernel=down_kernel),
                        scope="conv_pool_%d" % i)
                else:
                    # downsample with max pool
                    if len(conv_kernel) == 2:
                        x = slim.max_pool2d(x, [2, 2])
                    elif len(conv_kernel) == 3:
                        x = slim.max_pool3d(x, [2, 2, 2])

        # build expansion path
        for i in range(n_levels-1, 0, -1):
            x = upconv_sequence(x, i, convs,
                                is_training=is_training,
                                n_convs=n_convs,
                                start_channels=start_channels,
                                conv_kernel=conv_kernel,
                                inverted=inverted,
                                dropout_rate=dropout_rate,
                                upscale_method=upscale_method)

        # final convolution to map to number of classes
        final_kernel = [1 for dim in conv_kernel]
        x = slim.conv2d(x, n_classes, final_kernel,
                        activation_fn=None,
                        normalizer_fn=None,
                        weights_initializer=init(
                            x, conv_kernel=final_kernel, init='he'),
                        scope="final_conv")

    return x


def output_from_predictions(predictions):
    """
    get output map from logits
    
    Args:
        predictions (tf.Tensor): logits, 
            of shape [batch_num, height, width, n_classes]
    
    Returns:
        tf.Tensor: predicted map, of shape [batch_num, height, width, 1]
    """
    output = tf.argmax(
        predictions,
        axis=-1
    )
    output = tf.expand_dims(
        output,
        dim=-1
    )
    # cast to same type as ground truth
    output = tf.cast(
        output,
        tf.uint8
    )
    return output

def resize3d(input_layer, factor, method):
    shape = input_layer.shape
    input_reshaped = tf.reshape(input_layer, [-1, shape[1], shape[2], shape[3]*shape[4]])
    rsz1_reshaped = tf.image.resize_images(input_reshaped, 
                                  [shape[1]*factor, shape[2]*factor], 
                                  method=method)

    rsz1 = tf.reshape(rsz1_reshaped, [-1, shape[1]*factor, shape[2]*factor, shape[3], shape[4]])
    rsz1_transposed = tf.transpose(rsz1, [0, 3, 2, 1, 4])
    rsz1_transposed_reshaped = tf.reshape(rsz1_transposed,
                                          [-1, shape[3], shape[2]*factor, shape[1]*factor*shape[4]])

    rsz2_transposed_reshaped = tf.image.resize_images(rsz1_transposed_reshaped, 
                                                      [shape[3]*factor, shape[2]*factor],
                                                      method=method)

    rsz2_transposed = tf.reshape(rsz2_transposed_reshaped,
                                 [-1, shape[3]*factor, shape[2]*factor, shape[1]*factor, shape[4]])
    rsz2 = tf.transpose(rsz2_transposed, [0, 3, 2, 1, 4])
    return rsz2
