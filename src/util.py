"""Summary
"""
import functools
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndi
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from contextlib import contextmanager
import imageio
from collections import deque
from itertools import product

def joint_apply(image, ground_truth, fn, do_map=False,
                wrap_python=False, wrap_rand=None):
    """
    apply augmentation function to both the image and the labels
    
    Args:
        image (tf.Tensor): tensor of images
        ground_truth (tf.Tensor): tensor of labels
        fn (function): function to apply
        do_map (bool, optional): if true, map to tensors along axis 0
        wrap_python (bool, optional): if true, wrap python function 
        	with tf.py_func
        wrap_rand (str, optional): type of wrapping to do in regards to random
        	state handling, one of 'state', for partially applying random state,
        	'seed' for partially applying seeds or None
    
    Returns:
        (tf.Tensor, tf.Tensor): image and label tensors
    """
    new_seed = np.random.randint(np.iinfo(np.int32).max)
    if wrap_rand == "state":
        if do_map:
            fn = wrap_random_state_map(fn)
            do_map = False

        random_state_1 = np.random.RandomState(new_seed)
        random_state_2 = np.random.RandomState(new_seed)
        fn_1 = functools.partial(fn, random_state=random_state_1)
        fn_2 = functools.partial(fn, random_state=random_state_2)
    elif wrap_rand == "seed":
        fn_1 = functools.partial(fn, seed=new_seed)
        fn_2 = functools.partial(fn, seed=new_seed)
    else:
        fn_1 = fn
        fn_2 = fn

    if wrap_python:
        fn_image = wrap_python_fn(fn_1, image.dtype)
        fn_gt = wrap_python_fn(fn_2, ground_truth.dtype)
    else:
        fn_image = fn_1
        fn_gt = fn_2

    # tf.set_random_seed(new_seed)
    if do_map:
        image = tf.map_fn(fn_image, image)
    else:
        image = fn_image(image)

    # tf.set_random_seed(new_seed)
    if do_map:
        ground_truth = tf.map_fn(fn_gt, ground_truth)
    else:
        ground_truth = fn_gt(ground_truth)

    return image, ground_truth


def wrap_random_state_map(fn):
    """
    wrap function to map to multiple elements, sharing the
    same random state
    
    Args:
        fn (function): function to wrap
    
    Returns:
        function: wrapped function
    """
    def inner_wrap_random_state_map(x, random_state):
        res = []
        for i in range(x.shape[0]):
            res.append(fn(x[i], random_state))
        return np.array(res)

    return inner_wrap_random_state_map


def wrap_python_fn(fn, Tout):
    """
    wrap python function, retaining its shape in the graph
    
    Args:
        fn (function): funtion to wrap
        Tout (tf.dtype): datatype of output
    
    Returns:
        tf.Tensor: op containing wrapped function
    """
    def inner_wrap_python_fn(x):
        return tf.reshape(tf.py_func(fn, [x], Tout), tf.shape(x))

    return inner_wrap_python_fn

def image_flip(flip_prob, axes):
    def inner(x, seed):
        val = tf.random_uniform(
            [],
            minval=0,
            maxval=1,
            dtype=tf.float32,
            seed=seed,
            name=None
        )
        ret = tf.cond(val < flip_prob, lambda : tf.reverse(x, axes), lambda : x)
        return ret
    return inner


def data_augmentation(image, ground_truth, noise_amount, flip_prob,
                      seed=42):
    """
    apply data augmentation to image and ground truth
    
    Args:
        image (tf.Tensor): images, 
        	of shape [num_batch, height, width, n_channels]
        ground_truth (tf.Tensor): images, of shape [num_batch, height, width, 1]
        noise_amount (float): amount of noise to apply
        max_rot (float): maximum rotation to apply, in angles
        elastic_alpha (float): alpha value for elastic deformation
        elastic_sigma (float): alpha value for elastic deformation
        max_reduction (float): max reduction in image size, 
        	0 retains size, .5 is half size
        seed (int, optional): random seed to use
    
    Returns:
        (tf.Tensor, tf.Tensor): tensors of augmented image and ground truth
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # random flips
    image, ground_truth = joint_apply(
        image, ground_truth,
        image_flip(flip_prob, [2]),
        do_map=True, wrap_rand="seed"
    )

    # random noise
    image = image * tf.random_normal(
        shape=tf.shape(image),
        mean=1.,
        stddev=noise_amount,
        dtype=tf.float32
    )

    return image, ground_truth


def merge_arrays(l1, l2):
    """
    merge two arrays
    
    Args:
        l1 (np.array): first array
        l2 (np.array): second array
    
    Returns:
        list: merged array
    """
    l1 = np.split(l1, l1.shape[0], axis=0)
    l2 = np.split(l2, l2.shape[0], axis=0)
    return [val[0] for pair in zip(l1, l2) for val in pair]


def image_grid(el_list, img_fn, label_fn, per_col, mul):
    """
    make a grid of images
    
    Args:
        el_list (list): list of elements to display
        img_fn (function): function mapping elements to images
        label_fn (function): function mapping elements to labels
        per_col (int): number of images per column
        mul (float): size multiplier
    """
    n_images = len(el_list)
    fig = plt.figure(figsize=(per_col * mul, np.ceil(n_images / per_col) * mul))
    for index, el in enumerate(el_list):
        img = img_fn(el)
        ax = plt.subplot(np.ceil(n_images / per_col), per_col, 1 + index)
        ax.set_xlabel(label_fn(el))
        imgplot = plt.imshow(img)
    fig.tight_layout()
    plt.show()


def class_weights_from_freqs(frequencies, p=None, wp=None):
    """
    calculate weight for each class from relative frequencies
    
    Args:
        frequencies (list): relative frequency of each class, 
        	of shape [n_classes]

    Returns:
        np.array: weight for each class, of shape [n_classes]
    """
    if p is None:
        num_classes = len(frequencies)
        class_weights = 1 / np.array(frequencies, np.float32) / num_classes
    else:
        num_classes = len(frequencies[0])
        # interpolate the class weights, according to the probability p
        class_weights_0 = 1 / np.array(frequencies[0], np.float32) / num_classes
        class_weights_1 = 1 / np.array(frequencies[1], np.float32) / num_classes
        class_weights = (p * class_weights_0) + ((1 - p) * class_weights_1)
        class_weights = (wp * class_weights) + (1 - wp)

    return class_weights


def divide_by_max(x):
    """
    divide tensor by its maximum
    
    Args:
        x (tf.Tensor): input tensor
    
    Returns:
        tf.Tensor: tensor divided by its max value
    """
    return x / tf.reduce_max(x)


def tic():
    global tic_
    tic_ = time.time()

def toc():
    global tic_
    toc_ = time.time()
    print('Elapsed time: %fs' % (toc_ - tic_))

@contextmanager
def tictoc():
    tic()
    yield
    toc()

# https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume, grid_size=(1,5)):
    remove_keymap_conflicts({'j', 'k'})

    fig = plt.figure()
    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
                wspace=0, hspace=0)

    fig.volume = volume
    fig.index = volume.shape[0] // 2

    for c in range(volume.shape[-1]):
        ax = fig.add_subplot(*grid_size, c + 1)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.imshow(volume[fig.index, ..., c])
    
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    if event.key == 'j':
        previous_slice(fig)
    elif event.key == 'k':
        next_slice(fig)
    fig.canvas.draw()

def previous_slice(fig):
    volume = fig.volume
    fig.index = (fig.index - 1) % volume.shape[0]
    for c, ax in enumerate(fig.axes):
        if len(ax.images) > 0:
            ax.images[0].set_array(volume[fig.index, ..., c-1])
    
def next_slice(fig):
    volume = fig.volume
    fig.index = (fig.index + 1) % volume.shape[0]
    for c, ax in enumerate(fig.axes):
        if len(ax.images) > 0:
            ax.images[0].set_array(volume[fig.index, ..., c-1])

def to_gif(array, path, fps=60):
    array = float_to_int(array, dtype=np.uint8)
    imageio.mimsave(
        path + '.gif', 
        [array[i] for i in range(len(array))], 
        fps=fps
    )


def float_to_int(img, dtype=np.uint16):
    return (img * (np.iinfo(dtype).max)).astype(dtype)


class StoppingCriterion(object):
    def __init__(self, maxlen=500, patience=500):
        self.collection = deque([0]*maxlen, maxlen=maxlen)
        self.average = 0.
        self.min_average = 0.
        self.patience = patience
        self.patience_counter = 0
        self.len = maxlen
        self.should_stop = False
        
    def update(self, acc):
        self.collection.append(acc)
        self.average += (acc - self.collection[0]) / self.len
        if self.average > self.min_average:
            self.min_average = self.average
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        self.should_stop = (self.patience_counter > self.patience)
    
def map_to_color(img):
    colors = [
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 255]
    ]
    labels = np.zeros(tuple(img.shape) + (3,))
    for i in product(*[range(l) for l in img.shape]):
        labels[i] = colors[img[i]]
        
    return labels