"""
Metric definitions
"""
import tensorflow as tf
from functools import partial


def image_summaries(images, ground_truth, logits, output):
    """
    Add summaries for model inputs and ouputs
    
    Args:
        images (tf.Tensor): input images, 
            of shape [batch_num, height, width, n_channels]
        ground_truth (tf.Tensor): ground truth tensor, 
            of shape [batch_num, height, width, 1]
        logits (tf.Tensor): prediction map of model, 
            of shape [batch_num, height, width, n_channels]
        output (tf.Tensor): output map of network, 
            of shape [batch_num, height, width, 1]
    
    """
    n_slices = images.get_shape()[1].value
    n_classes = logits.get_shape()[-1].value

    color_val = int((256 / (n_classes - 1)) - 1)

    for slice in [n_slices // 2]:
        
        tf.summary.image(
            "input_slice_%d" % slice,
            images[:, slice] * 255,
            max_outputs=1,
            collections=None,
            family=None
        )

        # both ground_truth and output have values in range [0, 2]
        # multiply by 127 to map them to black, gray and white
        tf.summary.image(
            "gt_slice_%d" % slice,
            ground_truth[:, slice] * color_val,
            max_outputs=1,
            collections=None,
            family=None
        )

        tf.summary.image(
            "output_slice_%d" % slice,
            output[:, slice] * color_val,
            max_outputs=1,
            collections=None,
            family=None
        )

        # map values of each channel to range [0, 255]
        for c in range(n_classes):
            tf.summary.image(
                "logit_slice_%d_class_%d" % (slice, c),
                tf.expand_dims(logits[:, slice, ..., c] * 255, -1),
                max_outputs=1,
                collections=None,
                family=None
            )

            
def label_to_one_hot(output, ground_truth, n_classes=5):
    """
    Avoiding boilerplate code
    """
    gts_squeeze = tf.cast(tf.squeeze(ground_truth, -1), tf.int32)
    output_squeeze = tf.cast(tf.squeeze(output, -1), tf.int32)
    gts_one_hot = tf.one_hot(gts_squeeze, n_classes)
    output_one_hot = tf.one_hot(output_squeeze, n_classes)
    return output_one_hot, gts_one_hot


def indexing_and_sum(output, ground_truth, index_list, axis=-1):
    """
    Avoiding boilerplate code
    """
    result_output = tf.reduce_sum(tf.gather(output, index_list, axis=axis), axis=axis)
    result_ground_truth = tf.reduce_sum(tf.gather(ground_truth, index_list, axis=axis), axis=axis)
    return result_output, result_ground_truth


def dice_wrapper(output, ground_truth, classes_to_sum=[], smooth=1e-6):
    """
    Avoiding boilerplate code
    """
    output_one_hot, gts_one_hot = label_to_one_hot(output, ground_truth)
    output, ground_truth = indexing_and_sum(output_one_hot, gts_one_hot, classes_to_sum)
    intersection = tf.reduce_sum(output * ground_truth, axis=[1, 2, 3])
    union = tf.reduce_sum(output + ground_truth, axis=[1, 2, 3])
    result = (2 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(result)

dice_WT = partial(dice_wrapper, classes_to_sum=[1, 2, 3, 4])
dice_CT = partial(dice_wrapper, classes_to_sum=[1, 3, 4])
dice_ET = partial(dice_wrapper, classes_to_sum=[4])

def dice(output, ground_truth):
    return (dice_WT(output, ground_truth), 
            dice_CT(output, ground_truth), 
            dice_ET(output, ground_truth))



def sensitivity_wrapper(output, ground_truth, classes_to_sum=[], smooth=1e-6):
    """
    Avoiding boilerplate code
    """
    output_one_hot, gts_one_hot = label_to_one_hot(output, ground_truth)
    output, ground_truth = indexing_and_sum(output_one_hot, gts_one_hot, classes_to_sum)
    true_positive = tf.reduce_sum(output * ground_truth, axis=[1, 2, 3])
    false_negative = tf.reduce_sum((1 - output) * ground_truth, axis=[1, 2, 3])
    result = (true_positive + smooth) / (true_positive + false_negative + smooth)
    return tf.reduce_mean(result)

sensitivity_WT = partial(sensitivity_wrapper, classes_to_sum=[1, 2, 3, 4])
sensitivity_CT = partial(sensitivity_wrapper, classes_to_sum=[1, 3, 4])
sensitivity_ET = partial(sensitivity_wrapper, classes_to_sum=[4])

def sensitivity(output, ground_truth):
    return (sensitivity_WT(output, ground_truth), 
            sensitivity_CT(output, ground_truth), 
            sensitivity_ET(output, ground_truth))


def specifity_wrapper(output, ground_truth, classes_to_sum=[], smooth=1e-6):
    """
    Avoiding boilerplate code
    """
    output_one_hot, gts_one_hot = label_to_one_hot(output, ground_truth)
    output, ground_truth = indexing_and_sum(output_one_hot, gts_one_hot, classes_to_sum)
    true_negative = tf.reduce_sum((1 - output) * (1 - ground_truth), axis=[1, 2, 3])
    false_positive = tf.reduce_sum(output * (1 - ground_truth), axis=[1, 2, 3])
    result = (true_negative + smooth) / (true_negative + false_positive + smooth)
    return tf.reduce_mean(result)

specifity_WT = partial(specifity_wrapper, classes_to_sum=[1, 2, 3, 4])
specifity_CT = partial(specifity_wrapper, classes_to_sum=[1, 3, 4])
specifity_ET = partial(specifity_wrapper, classes_to_sum=[4])

def specifity(output, ground_truth):
    return (specifity_WT(output, ground_truth), 
            specifity_CT(output, ground_truth), 
            specifity_ET(output, ground_truth))


def accuracy(output, ground_truth, n_classes=5, smooth=1e-6):
    """
    Get per pixel accuracy
    
    Args:
        ground_truth (tf.Tensor): ground truth tensor, 
            of shape [batch_num, height, width, 1]
        output (tf.Tensor): output map of network,
            of shape [batch_num, height, width, 1]
    
    Returns:
        tf.Tensor: average pixel accuracy over all examples, of shape []
    """
  
    equal = tf.cast(tf.equal(output, ground_truth), tf.int32)
    acc = tf.reduce_sum(equal, axis=[1,2,3]) / tf.reduce_prod(equal.shape[1:])
    acc = tf.reduce_mean(acc)
    return acc