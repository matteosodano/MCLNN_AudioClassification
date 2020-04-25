import keras
import glob
import os
import numpy as np
from keras.models import Model
import matplotlib.cm as cm
import numpy as np
import numpy.ma as ma
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt



import keras
import glob
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras import callbacks
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as pyplt
import matplotlib
import matplotlib.pyplot as plt


    


class DirectoryHouseKeepingCallback(keras.callbacks.Callback):
    def __init__(self, filepath):
        self.learnedweightpath = filepath

    def on_epoch_end(self, epoch, logs={}):
        weightList = glob.glob(os.path.join(self.learnedweightpath, "*.hdf5"))
        weightList.sort(key=os.path.getmtime)
        if len(weightList) > 60:
            os.remove(weightList[0])


def prepare_callbacks(configuration, fold_weights_path, data_loader):
    callback_list = []

    # remote_callback = callbacks.RemoteMonitor(root='http://localhost:9000')
    # callback_list.append(remote_callback)

    # early stopping
    early_stopping_callback = callbacks.EarlyStopping(monitor=configuration.STOPPING_CRITERION,
                                                      patience=configuration.WAIT_COUNT,
                                                      verbose=0,
                                                      mode='auto')
    callback_list.append(early_stopping_callback)

    # save weights at the end of epoch
    weights_file_name_format = 'weights.epoch{epoch:02d}-val_loss{val_loss:.2f}-val_acc{val_acc:.4f}.hdf5'
    checkpoint_callback = ModelCheckpoint(os.path.join(fold_weights_path, weights_file_name_format),
                                          monitor='val_loss', verbose=0,
                                          save_best_only=False, mode='auto')
    callback_list.append(checkpoint_callback)

    # free space of stored weights of early epochs
    directory_house_keeping_callback = DirectoryHouseKeepingCallback(fold_weights_path)
    callback_list.append(directory_house_keeping_callback)



    return callback_list
