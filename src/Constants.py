"""
Module containing global constants.
Although most values cannot be made public, the file is shared to indicate which constants are used.
"""

from visualizer import Visualizer

NAME = "***"  # str
TESTNAME = "***"  # str
GPU = "0, 1"  # str; '0' for solely using GPU 0 and '0,1' to distribute training across GPU 0 and 1 (if possible)
CODE_TESTING = False  # bool; set to True for quick code testing -> 2 epochs with 1 example, results not saved
PHASE = 2  # int; 1 for training, 2 for testing (other phases not included in public codebase)

LOSS = "***"  # str; loss function name
MODEL = "***"  # str; model name
LR_RESUNET = None  # int; initial LR
NUM_WORKERS = None  # int; number of workers
MAX_LABEL = (
    2  # value of maximum label (2 for tl/fl segmentation, 3 for tl/fl/flt segmentation)
)
INP_CH = 1  # int; no. of input channels
OUTP_CH = MAX_LABEL + 1  # int; no. of output channels
PREV_WEIGHTS = False  # bool; Start training with saved weights (make sure to use same number of GPUs)
SAVE_PREDS = False  # bool; whether to save the predictions
L2 = None  # float; L2 regularization (weight decay) lambda value
AUGM = (
    True,
    None,
)  # tuple(bool, float); (whether to perform data augmentation, percentage of augmented examples)

if CODE_TESTING:
    TOTAL_EPOCH = 2
    NAME = "TEST"
else:
    TOTAL_EPOCH = None  # int
NUM_EARLY_STOP = (
    None  # int; no. of epochs after which to potentially perform early stopping
)

MAIN_DIR = "***"  # str; main directory path
DIR_DATA = "***"  # str; data directory path
ROOT_ORG = DIR_DATA + "PATH/"
DATASET = "***"  # str; dataset folder name
ROOT_TRAIN = DIR_DATA + DATASET + "/train/"
ROOT_VAL = DIR_DATA + DATASET + "/val/"
ROOT_TEST = DIR_DATA + DATASET + "/test/"

NB_SLICES = None  # int; no. of slices per slab
NB_SLABS = None  # int; no. of random slabs per scan


def log_vars():
    """Log all constants and values to visualizer."""
    show = Visualizer()
    show.Log("*** Constants:", printing=False)
    for name, value in globals().copy().items():
        show.Log(name + ": " + str(value), printing=False)
    show.Log("***\n***", printing=False)
