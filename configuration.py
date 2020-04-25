"""
MCLNN configuration
"""
import os


class Configuration:
    USE_PRETRAINED_WEIGHTS = False  # True or False - no training is initiated (pre-trained weights are used)

    TRAIN_SEGMENT_INDEX = 500  # train segment index to plot during training
    TEST_SEGMENT_INDEX = 500  # test segment index to plot during training
    VALIDATION_SEGMENT_INDEX = 500  # validation segment index to plot during training

    NB_EPOCH = 15 #2000  # maximum number of epochs
    WAIT_COUNT = 8#5  # early stopping count
    LEARNING_RATE = 0.0001
    SPLIT_COUNT = 3  # training/testing/validation splits
    TRAIN_FOLD_NAME = 'train'
    TEST_FOLD_NAME = 'test'
    VALIDATION_FOLD_NAME = 'validation'
    STOPPING_CRITERION = 'val_acc'  # 'val_acc' or 'val_loss'

    TENSOR_BOARD_MODEL_PATH = 'D:/tesnsorflow_log'



class GTZAN(Configuration):
    DATASET_NAME = 'gtzan'
    CROSS_VALIDATION_FOLDS_COUNT = 10
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    PARENT_PATH = 'I:/dataset-gtzan'

    COMMON_PATH_NAME = os.path.join(PARENT_PATH, DATASET_NAME + '_folds_' + str(CROSS_VALIDATION_FOLDS_COUNT))
    INDEX_PATH = COMMON_PATH_NAME + '_index'
    STANDARDIZATION_PATH = COMMON_PATH_NAME + '_standardization'
    ALL_FOLDS_WEIGHTS_PATH = COMMON_PATH_NAME + '_weights'

    DATASET_FILE_PATH = os.path.join(PARENT_PATH,
                                     'gtzanSpecmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=23_FN=600_30secs.hdf5')

    STEP_SIZE = 1  # overlap between segments is q minus step_size
    BATCH_SIZE = 600  # the samples in a mini-batch
    NB_CLASSES = 10  # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.35, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 200, 50, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 1  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [4, 4]  # the order for each layer
    # Type (MCLNN/CLNN) of each layer before pooling
    LAYER_IS_MASKED = [True, True]  # True: MCLNN, False: CLNN (Bandwidth and Overlap are ignored in this case)
    MASK_BANDWIDTH = [40, 10]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-10, 3]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 10  # the k extra frames

    CLASS_NAMES = ['Bl', 'Cl', 'Co', 'Di', 'Hi', 'Ja', 'Me', 'Po', 'Re', 'Ro']


class ISMIR2004(Configuration):
    DATASET_NAME = 'ismir2004'
    CROSS_VALIDATION_FOLDS_COUNT = 10
    INITIAL_FOLD_ID = 0  # the initial fold to start with. This should be zero unless you want to start from another fold
    PARENT_PATH = 'I:/dataset-ismir2004'

    COMMON_PATH_NAME = os.path.join(PARENT_PATH, DATASET_NAME + '_folds_' + str(CROSS_VALIDATION_FOLDS_COUNT))
    INDEX_PATH = COMMON_PATH_NAME + '_index'
    STANDARDIZATION_PATH = COMMON_PATH_NAME + '_standardization'
    ALL_FOLDS_WEIGHTS_PATH = COMMON_PATH_NAME + '_weights'

    DATASET_FILE_PATH = os.path.join(PARENT_PATH,
                                     'ismir2004Specmeln_mels=256_nfft=2048_hoplength=1024_fmax=NIL_22050hzsampling_FF=600_FN=600_30secs.hdf5')

    STEP_SIZE = 1#1  # overlap between segments is q minus step_size
    BATCH_SIZE = 600# 600  # the samples in a mini-batch
    NB_CLASSES = 6  # the number of classes to classify
    DROPOUT = [0.01, 0.35, 0.35, 0.1]  # dropout at the input of each layer
    HIDDEN_NODES_LIST = [220, 200, 50, NB_CLASSES]  # hidden nodes for each layer
    WEIGHT_INITIALIZATION = ['he_normal', 'he_normal', 'glorot_uniform', 'glorot_uniform']

    # Model layers
    MCLNN_LAYER_COUNT = 2  # number of MCLNN layers
    DENSE_LAYER_COUNT = 1  # number of dense layers

    # MCLNN hyperparameters
    LAYERS_ORDER_LIST = [4, 4]  # the order for each layer
    # Type (MCLNN/CLNN) of each layer before pooling
    LAYER_IS_MASKED = [True, True]  # True: MCLNN, False: CLNN (Bandwidth and Overlap are ignored in this case)
    MASK_BANDWIDTH = [40, 10]  # the consecutive features enabled at the input for each layer
    MASK_OVERLAP = [-10, 3]  # the overlap of observation between a hidden node and another for each layer
    EXTRA_FRAMES = 10#à######10  # the k extra frames

    CLASS_NAMES = ['Cl', 'El', 'Ja', 'Me', 'Po', 'Wo']


