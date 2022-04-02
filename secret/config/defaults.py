from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCH = 'resnet50'

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True
_C.MODEL.PART_DETACH = False

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
_C.MODEL.LOSSES.CE.EPSILON = 0.1

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()

_C.MEAN_TEACH = CN()
_C.MEAN_TEACH.CE_SOFT_WRIGHT = 0.5
_C.MEAN_TEACH.TRI_SOFT_WRIGHT = 0.8
_C.MEAN_TEACH.ALPHA = 0.999

_C.CLUSTER = CN()
_C.CLUSTER.K1 = 30
_C.CLUSTER.K2 = 6
_C.CLUSTER.EPS = 0.600
_C.CLUSTER.REFINE_K = 0.4
# -----------------------------------------------------------------------------
# INPU
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING = 10

# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.MEAN = [0.485, 0.456, 0.406]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

_C.DATASETS.SOURCE = "dukemtmc"

_C.DATASETS.TARGET = "market1501"

_C.DATASETS.DIR = "Data"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCES = 4
_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.BATCH_SIZE = 64
_C.DATALOADER.ITER_MODE = True
_C.DATALOADER.ITERS = 100

# ---------------------------------------------------------------------------- #
# OPTIM
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()
_C.OPTIM.OPT = 'adam'
_C.OPTIM.LR = 0.00035
_C.OPTIM.WEIGHT_DECAY = 5e-04
_C.OPTIM.MOMENTUM = 0.9

_C.OPTIM.SGD_DAMPENING = 0
_C.OPTIM.SGD_NESTEROV = False

_C.OPTIM.RMSPROP_ALPHA = 0.99

_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.999

# Multi-step learning rate options
_C.OPTIM.SCHED = "warmupmultisteplr"
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.STEPS = [40, 70]

_C.OPTIM.WARMUP_ITERS = 10
_C.OPTIM.WARMUP_FACTOR = 0.01
_C.OPTIM.WARMUP_METHOD = "linear"

_C.OPTIM.EPOCHS = 80

_C.TEST = CN()
_C.TEST.PRINT_PERIOD = 200

# Re-rank
_C.TEST.RERANK = CN()
_C.TEST.RERANK.ENABLED = False
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.MODE = "USL"
_C.OUTPUT_DIR = "log/test"
_C.RESUME = ""
_C.PRINT_PERIOD = 100
_C.SEED = 1
_C.GPU_Device = [0,1,2,3]

_C.CHECKPOING = CN()
_C.CHECKPOING.REMAIN_CLASSIFIER = True
_C.CHECKPOING.SAVE_STEP = [10]
_C.CHECKPOING.PRETRAIN_PATH = ''
_C.CHECKPOING.EVAL = ''

_C.CUDNN_BENCHMARK = True
