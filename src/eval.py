import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
matplotlib.rcParams.update({'font.size': 12, 'font.family': 'sans'})

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics

import numpy as np
from itertools import product
import math
import sys

import os

import common
from model import model, output_from_predictions
import data
import metrics


flags = tf.app.flags

flags.DEFINE_string('data_path', common.RECORD_PATH_EVAL,
                    'Path to evaluation data in .tfrecord format')

flags.DEFINE_string('checkpoint_dir', common.LOG_PATH_TRAIN,
                    'Directory with the model checkpoint data')

flags.DEFINE_integer('patch_size', common.PATCH_SIZE,
                    'Size of the patches')

flags.DEFINE_integer('overlap', common.OVERLAP,
                    'Third root of the overlap for image reconstruction')


common.common_flags(flags)


FLAGS = flags.FLAGS


def get_latest_k_ckps(k=common.NUM_MODELS, model_path=None):
    if model_path == None:
        path = '../log/train/'
        model_path = os.path.join(path, os.listdir(path)[0])
    return tf.train.get_checkpoint_state(model_path).all_model_checkpoint_paths[-k:]


def start_end_slice(start, end):
    return [slice(s,e) for s, e in zip(start, end)]


def get_patch_locations(patch_size=common.PATCH_SIZE, overlap=common.OVERLAP, img_size=common.ORIGINAL_IMAGE_SIZE_LIST):
    """
    real overlap is overlap**3
    """
    
    nx = int(overlap * math.ceil(img_size[0] / patch_size))
    ny = int(overlap * math.ceil(img_size[1] / patch_size))
    nz = int(overlap * math.ceil(img_size[2] / patch_size))
    x = np.rint(np.linspace(0, img_size[0] - patch_size, num=nx)).astype(np.int32)
    y = np.rint(np.linspace(0, img_size[1] - patch_size, num=ny)).astype(np.int32)
    z = np.rint(np.linspace(0, img_size[2] - patch_size, num=nz)).astype(np.int32)
    
    x_end = [i + patch_size for i in x]
    y_end = [i + patch_size for i in y]
    z_end = [i + patch_size for i in z]
    
    return zip(product(x,y,z),product(x_end,y_end,z_end))
    

def img_to_patches(img):
    start_end_iter = get_patch_locations()
    rets = []
    for start,end in start_end_iter:
        patches = img[start_end_slice(start, end)]
        rets.append(patches)

    return np.stack(rets, axis=0)

def patches_to_images(patches, img_size=common.ORIGINAL_IMAGE_SIZE_LIST):
    patch_size = patches.shape[1]
    last_dim = [patches.shape[-1]]
    zrs_total = np.zeros(img_size + last_dim, dtype = np.float32)
    zrs_count = np.zeros(img_size + last_dim, dtype = np.float32)
    start_end_iter = get_patch_locations()
    for c,(start, end) in enumerate(start_end_iter):
        paddings = [*[[start[i], img_size[i] - end[i]] for i in range(3)], [0,0]]
        zrs_total += tf.pad(patches[c], paddings)
        zrs_count += tf.pad(tf.ones([patch_size]*3 + last_dim), paddings)
    zrs_count = tf.maximum(1., zrs_count)
    return zrs_total / zrs_count
  
def save_fig(dices_for_plot):
    bins = np.linspace(0,1,11)
    fig, ax = plt.subplots()
    ax.hist(dices_for_plot, bins, color=['blue', 'red', 'cyan'], rwidth=0.3, align='left')
    ax.set_xticks(bins)
    #ax.set_yticks([5,10,15,20])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Dice Scores', fontsize=14)
    ax.set_ylabel('Sample Number', fontsize=14)
    ax.legend(["Whole Tumor (N = 55)", "Core Tumor (N = 55)", "Enhancing Tumor (N = 55)"])
    plt.savefig('../imgs/Histogram.svg')

def main(args):
    tf.reset_default_graph()
    
    #pre stuff
    ckp_paths = get_latest_k_ckps(k=common.NUM_MODELS, model_path=None)
    dataset, metadata = data.get_dataset(FLAGS.data_path)
    NUM_EXAMPLES = int(metadata.total/(155*240*240))
    n_classes = len(metadata.frequencies)

    images, ground_truth = data.load_batch(
        [dataset],
        image_size=common.ORIGINAL_IMAGE_SIZE,
        input_slices=common.ORIGINAL_IMAGE_SIZE[0], 
        input_channels=4,
        batch_size=1,
        is_training=False
    )

    image_patches = img_to_patches(images[0])
    patches_list = [image_patches[i] for i in range(image_patches.shape[0])]


    PATCHES_PER_IMAGE = len(patches_list)
    print('patches per image:', PATCHES_PER_IMAGE)
    
    # patches --> queue
    q = tf.FIFOQueue(capacity=PATCHES_PER_IMAGE, dtypes=tf.float32, shapes=[common.PATCH_SIZE]*3 + [4])
    enqueue_op = q.enqueue_many(tf.stack(patches_list))
    patch = q.dequeue()
    
    # queue --> tf.train.batch
    tbatch = tf.train.batch([patch], common.BATCH_SIZE)
    
    # tf.train.batch --> models
    predictions = []
    for i in range(common.NUM_MODELS):
        with tf.variable_scope('%d_model' % i) as scope:

            prediction = model(
                tbatch,
                is_training=False,
                start_channels=FLAGS.start_channels,
                n_classes=n_classes,
                n_levels=FLAGS.n_levels,
                n_convs=FLAGS.n_convs,
                use_elu=FLAGS.use_elu,
                inverted=FLAGS.inverted,
                all_conv=FLAGS.all_conv,
                dropout_rate=0.,
                upscale_method=FLAGS.upscale_method,
                use_batch_norm=FLAGS.use_batch_norm,
                reg=0.
            )
            predictions.append(prediction)
            
    probs = tf.nn.softmax(predictions)
    
    #tf.stack not neccessary after softmax
    ensemble_average = tf.reduce_mean(tf.stack(probs, axis=0), axis=0)
    
    # results --> gathering queue
    recon_queue = tf.FIFOQueue(capacity=PATCHES_PER_IMAGE * 2, dtypes=tf.float32, shapes=[common.PATCH_SIZE]*3 + [5])
    enqueue_op_rec = recon_queue.enqueue_many(ensemble_average)
    recon_queue_r = tf.train.QueueRunner(recon_queue, [enqueue_op_rec] * 1)
    tf.train.add_queue_runner(recon_queue_r)
    full_image_patches = recon_queue.dequeue_many(PATCHES_PER_IMAGE)

    # reconstruction and prediction
    fully_reconstructed = patches_to_images(full_image_patches)
    fully_reconstructed = tf.expand_dims(fully_reconstructed, axis=0)
    fully_reconstructed = output_from_predictions(fully_reconstructed)
    
    
    # metrics
    dices = metrics.dice(fully_reconstructed, ground_truth)
    sensitivities = metrics.sensitivity(fully_reconstructed, ground_truth)
    specifities = metrics.specifity(fully_reconstructed, ground_truth)
    accuracy = metrics.accuracy(fully_reconstructed, ground_truth)
    
    # summaries
    tf.summary.image('image_slice_mod1', images[:,75,:,:,0:1])
    tf.summary.image('image_slice_mod2', images[:,75,:,:,1:2])
    tf.summary.image('image_slice_mod3', images[:,75,:,:,2:3])
    tf.summary.image('image_slice_mod4', images[:,75,:,:,3:4])
    tf.summary.image('gt_slice', ground_truth[:,75,:,:,:] * 63)
    tf.summary.image('prediction_slice', fully_reconstructed[:,75,:,:,:] * 63)
    
    tf.summary.scalar('dice_WT', dices[0])
    tf.summary.scalar('dice_CT', dices[1])
    tf.summary.scalar('dice_ET', dices[2])
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(common.LOG_PATH_TEST)
  

    def name_in_checkpoint(var):
        return var.op.name[2:]
    variables_to_restore = tf.contrib.slim.get_variables_to_restore()
    variables_to_restore_list = []
    for i in range(common.NUM_MODELS):
        variables_to_restore_list.append({name_in_checkpoint(var):var for var in variables_to_restore if var.op.name[0]==str(i)})


    with tf.Session() as sess:
        for i in range(common.NUM_MODELS):
            saver = tf.train.Saver(variables_to_restore_list[i])
            saver.restore(sess, ckp_paths[i])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        all_metrics_total = [0 for i in range(10)]
        dices_for_plot = [[], [], []]
        for n in range(NUM_EXAMPLES):
            print('Evaluating image %d of %d...\n' %(n+1, NUM_EXAMPLES))
            _, summary, *all_metrics = sess.run([enqueue_op, merged, *dices, *sensitivities, *specifities, accuracy])
            writer.add_summary(summary, n)
            print('Metrics of image %d' % (n+1))
            print('dice_metrics:', *all_metrics[0:3])
            print('sens_metrics:', *all_metrics[3:6])
            print('specs_metrics:', *all_metrics[6:9])
            print('accurcacy:', all_metrics[9])

            all_metrics_total = [x * n for x in all_metrics_total]
            all_metrics_total = [sum(x) / (n+1) for x in zip(all_metrics, all_metrics_total)]
            dices_for_plot[0].append(all_metrics[0])
            dices_for_plot[1].append(all_metrics[1])
            dices_for_plot[2].append(all_metrics[2])

            print('\nAveraged Metrics:')
            print('dice_metrics_avg:', *all_metrics_total[0:3])
            print('sens_metrics_avg:', *all_metrics_total[3:6])
            print('specs_metrics_avg:', *all_metrics_total[6:9])
            print('accurcacy_avg:', all_metrics_total[9])
            print('-----------------------\n')
        save_fig(dices_for_plot)
        sys.exit()
        coord.request_stop()
        coord.join(threads)
        
if __name__=='__main__':
    tf.app.run()
