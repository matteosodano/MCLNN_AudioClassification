import os

class GTZAN:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 800 #1000!!!!!!!
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = '/content/drive/My Drive/Colab Notebooks/ProgettoNeuralNetworks/gtzan/'
    DST_PATH = 'I:/dataset-gtzan'

    DATASET_NAME = "gtzan"
    # dataset standard file length of the GTZAN = 30 seconds
    DEFAULT_DURATION = "30secs"
    # at a sampling rate of 22050 sample per second and nfft 2048 overlap 1024 > 22050 * 30 sec / 1024 = 645 frames
    FIRST_FRAME_IN_SLICE = 23  # to avoid disruptions at the beginning
    FRAME_NUM = 600  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 256
    FFT_BINS = 2048
    HOP_LENGTH_IN_SAMPLES = 1024
    INCLUDE_DELTA = False

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0


class ISMIR2004:
    # file count for the dataset
    AUGMENTATION_VARIANTS_COUNT = 0
    DATASET_ORIGINAL_FILE_COUNT = 1169#1458
    TOTAL_EXPECTED_COUNT = DATASET_ORIGINAL_FILE_COUNT + DATASET_ORIGINAL_FILE_COUNT * AUGMENTATION_VARIANTS_COUNT
    SRC_PATH = '/content/drive/My Drive/Colab Notebooks/ProgettoNeuralNetworks/ISMIR_80/'
    DST_PATH = 'I:/dataset-ismir2004'

    DATASET_NAME = "ismir2004"
    # different length for each file in the ISMIR2004. we will extract the second 30secs from each file
    DEFAULT_DURATION = "30secs"
    # at a sampling rate of 22050 sample per second and nfft 2048 overlap 1024 > 22050 * 30 sec / 1024 = 645 frames
    FIRST_FRAME_IN_SLICE = 600#########600  # to avoid disruptions at the beginning
    FRAME_NUM = 600  # this is enough to avoid disruptions at the end of the clip
    MEL_FILTERS_COUNT = 256
    FFT_BINS = 2048
    HOP_LENGTH_IN_SAMPLES = 1024
    INCLUDE_DELTA = False

    PROCESSING_BATCH = 10
    SLEEP_TIME = 0

