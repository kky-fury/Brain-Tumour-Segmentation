"""
Loss definitions
"""
import tensorflow as tf
import numpy as np
import common
from util import class_weights_from_freqs

# [0.98, 0.005, 0.005, 0.005, 0.005]
def loss(logits, gts, loss="cross_entropy",
         frequencies=None, p=None, wp=None):
    """
    model loss

    Args:
        logits (tf.Tensor): prediction logits,
            of shape [batch_num, height, width, n_classes]
        gts (tf.Tensor): ground truth,
            of shape [batch_num, height, width, 1]
        loss (str, optional): type of loss to use,
            one of 'jaccard' or 'cross_entropy', defaults to 'cross_entropy'
        frequencies (list, optional): expected relative frequency of each class,
            of shape [n_classes]

    Returns:
        tf.Tensor: loss tensor, of shape []

    Raises:
        ValueError: invalid loss
    """
    if frequencies is not None:
        class_weights = class_weights_from_freqs(frequencies, p, wp)
    else:
        class_weights = None

    if loss == "cross_entropy":
        return cross_entropy_loss(
            logits, gts,
            class_weights,
            n_classes=5
        )

    elif loss == "jaccard":
        return jaccard_loss(
            logits, gts,
            class_weights,
            n_classes=5
        )
    elif loss == "dice":
        return dice_loss(
            logits, gts,
            class_weights,
            n_classes = 5
        )
    elif loss == "cross_entropy_normal":
        return cross_entropy_normal(
            logits, gts,
            n_classes = 5)
    else:
        raise ValueError("Invalid loss")


def cross_entropy_loss(logits, gts, class_weights, n_classes=5):
    """
    get the cross entropy loss

    Args:
        logits (tf.Tensor): prediction logits, of shape
            [batch_num, height, width, n_classes]
        gts (tf.Tensor): ground truth, of shape [batch_num, height, width, 1]
        class_weights (np.array): weight of each class, of shape [n_classes]
        n_classes (int, optional): number of classes

    Returns:
        tf.Tensor: cross entropy loss tensor, of shape []
    """
     # reshape logits to [batch_size, height*width*depth, n_classes]
    flat_logits = tf.reshape(
        logits, [-1, np.prod(logits.get_shape().as_list()[1:4]), n_classes])

    # reshape ground truths to [batch_size, height*width*depth]
    gts_flattened = tf.cast(tf.reshape(
        gts, [-1, np.prod(gts.get_shape().as_list()[1:4])]), tf.int32)


    cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=flat_logits,
        labels=gts_flattened
    )

    # average over pixels
    cross_entropies = cross_entropies / int(flat_logits.shape[1])

    # gather weight per pixel from ground truth pixel classes and class wieght
    weights = tf.gather(class_weights, gts_flattened)

    # multiply weight of each pixel
    weighted_cross_entropy = cross_entropies * weights

    # sum over pixels
    cross_entropy_sum = tf.reduce_sum(weighted_cross_entropy, axis=1)

    # average over examples
    cross_entropy_avg = tf.reduce_mean(cross_entropy_sum, axis=0)

    return cross_entropy_avg

def jaccard_loss(logits, gts, class_weights, n_classes=5, smooth=1e-6):
    """
    get the jaccard loss

    Args:
        logits (tf.Tensor): prediction logits, of shape
            [batch_num, height, width, n_classes]
        gts (tf.Tensor): ground truth, of shape [batch_num, height, width, 1]
        class_weights (np.array): weight of each class, of shape [n_classes]
        n_classes (int, optional): number of classes
        smooth (float, optional): smoothing constant, default to 1e-3

    Returns:
        tf.Tensor: jaccard loss tensor, of shape []
    """
    gts_squeeze = tf.cast(tf.squeeze(gts, -1), tf.int32)

    gts_one_hot = tf.one_hot(gts_squeeze, n_classes)

    preds = tf.nn.softmax(logits)

    gts_one_hot_flattened = tf.reshape(gts_one_hot, [-1, np.prod(gts_one_hot.get_shape().as_list()[1:4]), n_classes])
    preds_flattened = tf.reshape(preds, [-1, np.prod(preds.get_shape().as_list()[1:4]), n_classes])

    # sum intersections and sums over all pixels, for each class
    intersection = tf.reduce_sum(gts_one_hot_flattened * preds_flattened, axis=[1])
    sum_ = tf.reduce_sum(gts_one_hot_flattened + preds_flattened, axis=[1])

    jac = 1 - ((intersection + smooth) / (sum_ - intersection + smooth))

    # jac = jac * class_weights

    # average over classes
    jac = tf.reduce_mean(jac, axis=1)

    # average over examples
    jac = tf.reduce_mean(jac, axis=0)

    return jac

def dice_loss(logits, gts, class_weights, n_classes=5, smooth=1e-5):

    # cross_entropy_loss_val = cross_entropy_loss(logits, gts, class_weights, n_classes)
    jaccard_loss_ = jaccard_loss(logits,gts,class_weights, n_classes = 5)
    dice_loss = 2*jaccard_loss_/(1 + jaccard_loss_)
    return dice_loss

def cross_entropy_normal(logits, gts, n_classes = 5):
    flat_logits = tf.reshape(
        logits, [-1, np.prod(logits.get_shape().as_list()[1:4]), n_classes])
    gts_flattened = tf.cast(tf.reshape(
        gts, [-1, np.prod(gts.get_shape().as_list()[1:4])]), tf.int32)
    cross_entropie_loss= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=flat_logits,
        labels=gts_flattened
    ))
    return cross_entropie_loss