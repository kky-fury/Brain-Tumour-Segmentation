import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.learning import train_step
import time
import os
from model import model, output_from_predictions
from loss import loss
import metrics
import data
import common


flags = tf.app.flags

# Paths

flags.DEFINE_string('data_path', common.RECORD_PATH_TRAIN,
                    'Path to training data in .tfrecord format')

flags.DEFINE_string('log_dir', common.LOG_PATH_TRAIN,
                    'Directory to store log data')

flags.DEFINE_string('checkpoint_dir', common.LOG_PATH_TRAIN,
                    'Directory with the model checkpoint data')

common.common_flags(flags)

# Training hyperparameters

flags.DEFINE_integer('number_of_steps', common.NUMBER_OF_STEPS,
    'Number of training steps to perform')

flags.DEFINE_float('learning_rate', common.LEARNING_RATE, 'Learning rate value')

flags.DEFINE_float('reg', common.REGULARIZATION, 'L2 Regularization value')

flags.DEFINE_string('loss', common.LOSS, 'Loss to use for training')


# Architecture hyperparameters

flags.DEFINE_float('dropout_rate', common.DROPOUT_RATE,
    'Standard deviation for Gaussian dropout')

# Image augmentation

flags.DEFINE_integer('seed', common.SEED,
    'Seed to use for shuffling and data augmentation')

# Ensemble configuration
flags.DEFINE_integer('num_models', common.NUM_MODELS, 'Number of ensemble models to train')

FLAGS = flags.FLAGS


graph = tf.Graph()
with graph.as_default():
    dataset, metadata = data.get_dataset(common.RECORD_PATH_TRAIN)
    dataset_unbiased, metadata_unbiased = data.get_dataset(common.RECORD_PATH_TRAIN_UNBIASED)
    prob_biased = tf.Variable(1.0, name='prob_biased', trainable=False)
    weight_perc = tf.Variable(1.0, name='weight_perc', trainable=False)
    tf.summary.scalar("prob_biased", prob_biased)
    tf.summary.scalar("weight_perc", weight_perc)
    prob_less_zero = tf.less(prob_biased - common.SAMPLING_PROBABILITY_DELTA, 0)
    wp_less_zero = tf.less(weight_perc - common.SAMPLING_PROBABILITY_DELTA, 0)
    prob_update = tf.assign(prob_biased, tf.cond(prob_less_zero, lambda: 0.0, lambda: (prob_biased - common.SAMPLING_PROBABILITY_DELTA)))
    weight_perc_update = tf.assign(weight_perc, tf.cond(tf.logical_and(prob_less_zero, wp_less_zero), lambda: (weight_perc - common.WEIGHT_PERC_DELTA), lambda: 1.0))
    dataset_val, metadata_val = data.get_dataset(common.RECORD_PATH_VAL)
    dataset_val_unbiased, metadata_val_unbiased = data.get_dataset(common.RECORD_PATH_VAL_UNBIASED)

    images, ground_truth = data.load_batch(
        [dataset, dataset_unbiased],
        image_size=(common.IMAGE_SIZE, common.IMAGE_SIZE, common.IMAGE_SIZE),
        input_slices=common.IMAGE_SIZE,
        input_channels=4,
        batch_size=FLAGS.batch_size,
        is_training=True,
        noise_amount=.05, flip_prob=.5,
        seed=FLAGS.seed,
        prob_datasets=prob_biased
    )

    image_val, ground_truth_val = data.load_batch(
        [dataset_val],
        image_size=(common.IMAGE_SIZE, common.IMAGE_SIZE, common.IMAGE_SIZE),
        input_slices=common.IMAGE_SIZE,
        input_channels=4,
        batch_size=FLAGS.batch_size,
        is_training=False,
        seed=FLAGS.seed
    )
    image_val_unbiased, ground_truth_val_unbiased = data.load_batch(
        [dataset_val_unbiased],
        image_size=(common.IMAGE_SIZE, common.IMAGE_SIZE, common.IMAGE_SIZE),
        input_slices=common.IMAGE_SIZE,
        input_channels=4,
        batch_size=FLAGS.batch_size,
        is_training=False,
        seed=FLAGS.seed
        )
    with tf.variable_scope("model") as scope:
        logits = model(
            images,
            is_training=True,
            start_channels=FLAGS.start_channels,
            n_classes=5,
            n_levels=FLAGS.n_levels,
            n_convs=FLAGS.n_convs,
            use_elu=FLAGS.use_elu,
            inverted=FLAGS.inverted,
            all_conv=FLAGS.all_conv,
            dropout_rate=FLAGS.dropout_rate,
            upscale_method=FLAGS.upscale_method,
            use_batch_norm=FLAGS.use_batch_norm,
            reg=FLAGS.reg
        )
        scope.reuse_variables()
        predictions = model(
            image_val,
            is_training=False,
            start_channels=FLAGS.start_channels,
            n_classes=5,
            n_levels=FLAGS.n_levels,
            n_convs=FLAGS.n_convs,
            use_elu=FLAGS.use_elu,
            inverted=FLAGS.inverted,
            all_conv=FLAGS.all_conv,
            dropout_rate=FLAGS.dropout_rate,
            upscale_method=FLAGS.upscale_method,
            use_batch_norm=FLAGS.use_batch_norm,
            reg=0.
        )
        predictions_unbiased = model(
            image_val_unbiased,
            is_training=False,
            start_channels=FLAGS.start_channels,
            n_classes=5,
            n_levels=FLAGS.n_levels,
            n_convs=FLAGS.n_convs,
            use_elu=FLAGS.use_elu,
            inverted=FLAGS.inverted,
            all_conv=FLAGS.all_conv,
            dropout_rate=FLAGS.dropout_rate,
            upscale_method=FLAGS.upscale_method,
            use_batch_norm=FLAGS.use_batch_norm,
            reg=0.
        )

    output = output_from_predictions(logits)
    predictions_val = output_from_predictions(predictions)
    predictions_val_unbiased = output_from_predictions(predictions_unbiased)
    l = loss(logits, ground_truth, loss=FLAGS.loss, frequencies=[metadata.frequencies, metadata_unbiased.frequencies], p=prob_biased, wp=weight_perc)
    tf.losses.add_loss(l)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate,
        name='opt'
    )
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=True
    )
    metrics.image_summaries(images, ground_truth, logits, output)
    train_acc = metrics.accuracy(output, ground_truth)
    train_dice_WT, train_dice_CT, train_dice_ET = metrics.dice(output, ground_truth)
    tf.summary.scalar("train_acc", train_acc)
    tf.summary.scalar("dice_complete_tumor", train_dice_WT)
    tf.summary.scalar("dice_core_tumor", train_dice_CT)
    tf.summary.scalar("dice_enhancing_tumor", train_dice_ET)
    val_accuracy = metrics.accuracy(predictions_val, ground_truth_val)
    dice_whole_tumor_val, dice_core_tumor_val, dice_enhancing_tumor_val = metrics.dice(predictions_val, ground_truth_val)
    tf.summary.scalar("val_accuracy", val_accuracy)
    tf.summary.scalar("dice_whole_tumor_val", dice_whole_tumor_val)
    tf.summary.scalar("dice_core_tumor_val", dice_core_tumor_val)
    tf.summary.scalar("dice_enhancing_tumor_val", dice_enhancing_tumor_val)
    val_accuracy_unbiased = metrics.accuracy(predictions_val_unbiased, ground_truth_val_unbiased)
    dice_whole_tumor_val_unbiased, dice_core_tumor_val_unbiased, dice_enhancing_tumor_val_unbiased = metrics.dice(predictions_val_unbiased, ground_truth_val_unbiased)
    tf.summary.scalar("val_accuracy_unbiased", val_accuracy_unbiased)
    tf.summary.scalar("dice_whole_tumor_val_unbiased", dice_whole_tumor_val_unbiased)
    tf.summary.scalar("dice_core_tumor_val_unbiased", dice_core_tumor_val_unbiased)
    tf.summary.scalar("dice_enhancing_tumor_val_unbiased", dice_enhancing_tumor_val_unbiased)

def train_step_fn(session, *args, **kwargs):
    total_loss, should_stop = train_step(session, *args, **kwargs)
    print('\rStep: {0}'.format(train_step_fn.step), end="")
    if train_step_fn.step % common.VALIDATION_CHECK == 0:
        accuracy = session.run(train_step_fn.val_accuracy)
        dice_WT = session.run(train_step_fn.dice_whole_tumor_val)
        dice_CT = session.run(train_step_fn.dice_core_tumor_val)
        dice_ET = session.run(train_step_fn.dice_enhancing_tumor_val)
        accuracy_unbiased = session.run(train_step_fn.val_accuracy_unbiased)
        dice_WT_unbiased = session.run(train_step_fn.dice_whole_tumor_val_unbiased)
        dice_CT_unbiased = session.run(train_step_fn.dice_core_tumor_val_unbiased)
        dice_ET_unbiased = session.run(train_step_fn.dice_enhancing_tumor_val_unbiased)
        # print('\nLoss: %.2f Val_Accuracy_ub: %.2f Dice_WT_ub: %f Dice_CT_ub: %f Dice_ET_ub: %f'%(total_loss, accuracy_unbiased, dice_WT_unbiased, dice_CT_unbiased, dice_ET_unbiased))
        print('\nLoss: %.2f Val_Accuracy: %.2f Dice_WT: %f Dice_CT: %f Dice_ET: %f Val_Accuracy_ub: %.2f Dice_WT_ub: %f Dice_CT_ub: %f Dice_ET_ub: %f'%(total_loss, accuracy, dice_WT, dice_CT, dice_ET, accuracy_unbiased, dice_WT_unbiased, dice_CT_unbiased, dice_ET_unbiased))
        if train_step_fn.step != 0:
            session.run(prob_update)
            session.run(weight_perc_update)

    train_step_fn.step +=1
    return [total_loss, should_stop]

train_step_fn.step = 0
train_step_fn.val_accuracy = val_accuracy
train_step_fn.dice_whole_tumor_val = dice_whole_tumor_val
train_step_fn.dice_core_tumor_val = dice_core_tumor_val
train_step_fn.dice_enhancing_tumor_val = dice_enhancing_tumor_val
train_step_fn.val_accuracy_unbiased = val_accuracy_unbiased
train_step_fn.dice_whole_tumor_val_unbiased = dice_whole_tumor_val_unbiased
train_step_fn.dice_core_tumor_val_unbiased = dice_core_tumor_val_unbiased
train_step_fn.dice_enhancing_tumor_val_unbiased = dice_enhancing_tumor_val_unbiased



slim.learning.train(
    train_op,
    logdir=os.path.join(FLAGS.log_dir, str(int(time.time()))),
    train_step_fn = train_step_fn,
    graph = graph,
    save_interval_secs=600,
    save_summaries_secs=5,
    log_every_n_steps=10,
    number_of_steps=int(common.NUMBER_OF_STEPS)
)
