"""
Image loading
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pickle
from util import data_augmentation, divide_by_max
import util
import common


class DatasetMetadata(object):
    def __init__(self, counts):
        self.total = np.sum(counts)
        self.frequencies = counts / self.total

    def save(self, pickle_path):
        with open(pickle_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def load(pickle_path):
        with open(pickle_path, "rb") as pickle_file:
            out = pickle.load(pickle_file)
        return out

def to_tf_records_3D(tf_record_path, data_gen, n_classes=5):
    """
    write data to .tfrecord
    
    Args:
        tf_record_path (path): path for saved .tfrecord file
        data_gen (generator): generator that yields data to save, in format
            image, label, where image has shape [depth, width, height, channels]
            label has shape [depth, width, height]
    """
    writer = tf.python_io.TFRecordWriter(tf_record_path + ".tfrecord")

    counts = np.zeros(n_classes, dtype=np.int64)

    for img, gt in data_gen:
        img_list = []
        gt_list = []

        img = util.float_to_int(img)

        for slice_index in range(img.shape[0]):
            #img_list_sub = []
            gt = gt.astype(np.uint8)
            _, gt_bytes = cv2.imencode(".png", gt[slice_index])
            gt_bytes = gt_bytes.tobytes()

            gt_list.append(gt_bytes)

            for channel_index in range(img.shape[-1]):
                # encode images
                _, img_bytes = cv2.imencode(".png", img[slice_index, ..., channel_index])
                img_bytes = img_bytes.tobytes()
                img_list.append(img_bytes)
            #img_list.append(img_list_sub)

        feature_dict = {
            'img': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=img_list)
            ),
            'gt': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=gt_list)
            )
        }

        feats = tf.train.Features(feature=feature_dict)

        example = tf.train.Example(features=feats)

        writer.write(example.SerializeToString())
        count = np.bincount(np.ravel(gt), minlength=n_classes)
        counts += count


    writer.close()
    DatasetMetadata(counts).save(tf_record_path + ".pickle")
    print("Class counts", counts)


def all_to_tf_records(input_path, output_path):
    """
    convert all data to .tfrecord
    
    Args:
        input_path (path, optional): path to inputs
    """
    names = ("train", "test", "leaderboard")
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    for name in names:
        input_full_path = os.path.join(input_path, name)
        output_full_path = os.path.join(record_path,  name + ".tfrecord")
        to_tf_records(output_full_path, read_data(input_full_path, verbose=True))


def get_dataset(path):
    """
    get tf.data.Dataset object with data
    
    Args:
        path (path): path to .tfrecord file
    
    Returns:
        (tf.data.Dataset, DatasetMetadata): dataset and metadata
    """
    dataset = tf.data.TFRecordDataset(path + ".tfrecord")
    metadata = DatasetMetadata.load(path + ".pickle")
    return dataset, metadata


def decode_and_resize_3D(input_dtype, output_size, output_dtype, label=False):
    """
    get a function that decodes, resizes and converts input labels
    
    Args:
        input_dtype (tf.dtype): type of input
        output_size (tuple): size of output
        output_dtype (tf.dtype): type to convert to
        label (bool): True if image
    
    Returns:
        function: function
    """
    def inner(x):
        x = tf.image.decode_png(
            x,
            channels=1,
            dtype=input_dtype
        )
        if label:
            x = tf.image.resize_nearest_neighbor(
                tf.expand_dims(x, 0),
                output_size
            )[0]
        else:
            x = tf.image.convert_image_dtype(
                x,
                output_dtype,
                saturate=False
            )

            x = tf.image.resize_bicubic(
                tf.expand_dims(x, 0),
                output_size
            )[0]

        return tf.cast(x, output_dtype)

    return inner


def decode_and_resize_all(encoded, input_slices, input_channels, input_dtype, output_size, output_dtype, label):

    image = tf.stack(encoded)
    image_shape = tf.shape(image)
    image = tf.reshape(image, (-1,))
    image = tf.map_fn(
        decode_and_resize_3D(
            input_dtype=input_dtype,
            output_size=output_size[1:3], 
            output_dtype=output_dtype, 
            label=label
        ),
        image,
        dtype=output_dtype,
        parallel_iterations=10,
        back_prop=False,
        swap_memory=False,
        infer_shape=True
    )
    new_shape = tf.stack((image_shape[0], input_slices, input_channels, *image.shape[1:3]), axis=0)
    image = tf.reshape(image, new_shape)
    image = tf.transpose(image, perm=[0, 1, 3, 4, 2])

    return image


def read_from_tfrecord_3D(tfrecord_serialized, output_image_size, 
                        input_slices, input_channels,
                        input_image_dtype=tf.uint16, input_gt_dtype=tf.uint8,
                        output_image_dtype=tf.float32, output_gt_dtype=tf.uint8):
    """
    decode and resize serialized examples
    
    Args:
        tfrecord_serialized (tf.Tensor): tensor with serialized examples
        output_image_size (tuple): target size for images and labels
        input_slices (int): number of slices of input
        input_channels (int): number of channels of input
        input_image_dtype (tf.dtype, optional): datatype of input images
        input_gt_dtype (tf.dtype, optional): datatpye of input labels
        output_image_dtype (tf.dtype, optional): datatype of output images
        output_gt_dtype (tf.dtype, optional): datatype of output labels
    Returns:
        (tf.Tensor, tf.Tensor): decoded image and ground truth tensors,
            of shape [n_batches, height, width, depth, channels]
    """
    tfrecord_features = tf.parse_example(
        tfrecord_serialized,
        features={
            'img': tf.FixedLenFeature([input_slices, input_channels], tf.string),
            'gt': tf.FixedLenFeature([input_slices], tf.string),
        }, name='features'
    )
    image = decode_and_resize_all(
        encoded=tfrecord_features['img'],
        input_slices=input_slices,
        input_channels=input_channels,
        input_dtype=input_image_dtype,
        output_size=output_image_size, 
        output_dtype=output_image_dtype, 
        label=False
    )
    ground_truth = decode_and_resize_all(
        encoded=tfrecord_features['gt'],
        input_slices=input_slices,
        input_channels=1,
        input_dtype=input_gt_dtype,
        output_size=output_image_size, 
        output_dtype=output_gt_dtype, 
        label=True
    )

    return image, ground_truth

def load_batch(datasets, image_size, input_slices, input_channels,
               n_epochs=None, batch_size=32, noise_amount=0, flip_prob=0,
               is_training=False, seed=None,
               prob_datasets=None):
    """
    get augmented batches of images and labels
    
    Args:
        datasets (tf.data.Dataset): datasets to load from
        image_size (tuple, optional): size of each image (height, width)
        input_slices (int): input_slices
        input_channels (int): input_channels
        n_epochs (int, optional): number of epochs to train for
        batch_size (int, optional): size of each batch
        is_training (bool, optional): true if training
        seed (int, optional): random seed to use
        prob_datasets (tf.Variable): variable with the datasets[0] probability
    
    Returns:
        (tf.Tensor, tf.Tensor): images and ground truth tensors,
            of shapes [num_batch, height, width, 1]
    """

    images = []
    ground_truths = []

    for i in range(len(datasets)):
        if is_training:
            datasets[i] = datasets[i].shuffle(buffer_size=256)

        datasets[i] = datasets[i].batch(batch_size)
        datasets[i] = datasets[i].repeat()
        iterator = datasets[i].make_one_shot_iterator()

        next_element = iterator.get_next()

        image, ground_truth = read_from_tfrecord_3D(
            next_element, 
            output_image_size=image_size, 
            input_slices=input_slices, 
            input_channels=input_channels
        )

        images.append(image)
        ground_truths.append(ground_truth)

    # FIXME: only supports 2 datasets in reality
    if prob_datasets is not None:
        dataset_cond = tf.less_equal(tf.reduce_sum(tf.random_uniform([1], 0, 1)), prob_datasets)
        image = tf.cond(dataset_cond, lambda: images[0], lambda: images[1])
        ground_truth = tf.cond(dataset_cond, lambda: ground_truths[0], lambda: ground_truths[1])
    else:
        image = images[0]
        ground_truth = ground_truths[0]

    if is_training:
        image, ground_truth = data_augmentation(
            image, ground_truth,
            noise_amount=noise_amount, flip_prob=.5,
            seed=seed
        )

    return image, ground_truth


if __name__ == "__main__":
    dataset, metadata = get_dataset(common.RECORD_PATH_TRAIN)
    images, ground_truth = load_batch(
        [dataset],
        image_size=(common.IMAGE_SIZE, common.IMAGE_SIZE, common.IMAGE_SIZE),
        input_slices=common.IMAGE_SIZE,
        input_channels=4,
        batch_size=100,
        is_training=False,
        noise_amount=.05, 
        flip_prob=.5,
        seed=None,
        prob_datasets=None
    )
    sample = 25
    depth = images.shape[1]
    with tf.Session() as sess:

        imgs, gts = sess.run([images, ground_truth])

        for i in range(imgs.shape[-1]):
            cv2.imwrite("../imgs/modality_%d.png" % (i), imgs[sample, depth//2, ..., i]*255)

        color_labels = util.map_to_color(gts[sample, depth//2, ..., 0])

        cv2.imwrite("../imgs/gt.png", color_labels)