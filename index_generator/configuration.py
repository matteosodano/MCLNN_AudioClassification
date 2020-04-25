"""
Configuration for MCLNN Index Generation
"""
import os

# FOLD_COUNT : The dataset is released into FOLD_COUNT-folds
# SHUFFLE_CATEGORY_CLIPS : enable shuffle the samples when assigning to the folds
# AUGMENTATION_VARIANTS_COUNT : data augmentation (https://towardsdatascience.com/data-augmentation-for-deep-learning-4fe21d1a4eb9) integer.
# It multiplies the number of samples of each class for achieving augmentation
# CLIP_COUNT_PER_CATEGORY_LIST : samples per category
# BATCH_SIZE_PER_FOLD_ASSIGNMENT : batch of samples assigned per fold in a single instance of assignment


class GTZAN:
    DATASET = 'gtzan'
    DST_PATH = 'I:/dataset-gtzan'
    FOLD_COUNT = 10
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = True
    AUGMENTATION_VARIANTS_COUNT = 0

    # samples per category following the category order: ('Bl', 'Cl', 'Co', 'Di', 'Hi', 'Ja', 'Me', 'Po', 'Re', 'Ro')
    CLIP_COUNT_PER_CATEGORY_LIST = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

class ISMIR2004:
    DATASET = 'ismir2004'
    DST_PATH = 'I:/dataset-ismir2004'
    FOLD_COUNT = 10
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    SHUFFLE_CATEGORY_CLIPS = True
    AUGMENTATION_VARIANTS_COUNT = 0

    # samples per category following the category order: ('Cl', 'El', 'Ja', 'Me', 'Po', 'Wo')
    CLIP_COUNT_PER_CATEGORY_LIST = [512, 184, 42, 72, 163, 196] #[640, 229, 52, 90, 203, 244]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1



