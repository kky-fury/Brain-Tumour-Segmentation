import os

INPUT_PATH = os.path.join("..", "data", "BRATS2015_Training")
RECORD_PATH = os.path.join("..", "data")

RECORD_PATH_TRAIN = os.path.join(RECORD_PATH, "data")
RECORD_PATH_TRAIN_UNBIASED = os.path.join(RECORD_PATH, "unbiased")
RECORD_PATH_EVAL = os.path.join(RECORD_PATH, "test")
RECORD_PATH_VAL = os.path.join(RECORD_PATH, "val")
RECORD_PATH_VAL_UNBIASED = os.path.join(RECORD_PATH, "val_unbiased")

LOG_PATH = os.path.join("..", "log")
LOG_PATH_EVAL = os.path.join(LOG_PATH, "eval")
LOG_PATH_TRAIN = os.path.join(LOG_PATH, "train")
LOG_PATH_TEST = os.path.join(LOG_PATH, "test")

BATCH_SIZE = 4
IMAGE_SIZE = 64
START_CHANNELS = 64
N_LEVELS = 5
N_CONVS = 2
USE_ELU = False
INVERTED = False
ALL_CONV = True
UPSCALE_METHOD = 'resize3d'
USE_BATCH_NORM = True

PATCHES_PER_IMAGE = 200
NUM_EPOCH = 5
ITER_PER_EPOCH=(164*PATCHES_PER_IMAGE)/BATCH_SIZE

VALIDATION_CHECK=100

NUMBER_OF_STEPS = NUM_EPOCH*ITER_PER_EPOCH
LEARNING_RATE = 1e-4
REGULARIZATION = 0.02
LOSS = 'cross_entropy'
DROPOUT_RATE = .2
SEED = None

ORIGINAL_IMAGE_SIZE = (155, 240, 240)
ORIGINAL_IMAGE_SIZE_LIST = [155, 240, 240]

EVAL_PATCH_SIZE = [IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE]
EVAL_STRIDE = [IMAGE_SIZE , IMAGE_SIZE , IMAGE_SIZE ]

NUM_MODELS = 3
PATCH_SIZE = 128
OVERLAP = 1

SAMPLING_PROBABILITY_DELTA = 0.11
WEIGHT_PERC_DELTA = 0.1

def common_flags(flags):
    """
    set flags common to training and evaluation

    Args:
        flags (tf.app.flags): flags
    """

    # batch parameters

    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size')

    flags.DEFINE_integer('image_size', IMAGE_SIZE, 'Target size for images')

    # architecture

    flags.DEFINE_integer('start_channels', START_CHANNELS,
    ('Number of channels of activation maps '
     'in first level of the u-net architecture'))

    flags.DEFINE_integer('n_levels', N_LEVELS,
        'Number of levels of the u-net architecture, (5 in original paper)')

    flags.DEFINE_integer('n_convs', N_CONVS,
        ('Number of convolutions to perform at each level, '
        'on each path (2 in original paper)'))

    flags.DEFINE_bool('use_elu', USE_ELU,
        'Use the ELU activation function, else use RELU')

    flags.DEFINE_bool('inverted', INVERTED,
        'Use inverted variant of u-net architecture')

    flags.DEFINE_bool('all_conv', ALL_CONV,
        'Replace max pool operations with convolutions')

    flags.DEFINE_string('upscale_method', UPSCALE_METHOD,
                        'Upscaling method to use')

    flags.DEFINE_bool('use_batch_norm', USE_BATCH_NORM,
        'Use batch norm')
