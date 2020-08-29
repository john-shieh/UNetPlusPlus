from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
import os
import keras
print("Keras = {}".format(keras.__version__))
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
from tqdm import tqdm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pylab
import sys
import math
import SimpleITK as sitk
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import shutil
from sklearn import metrics
import random
from random import shuffle
from keras.callbacks import LambdaCallback, TensorBoard
from glob import glob
from skimage.transform import resize
from optparse import OptionParser
from segmentation_models import Nestnet, Unet, Xnet
from helper_functions import *
from keras.utils import plot_model

ROOT_PATH = os.path.abspath(".")
DATA_DIR = os.path.join(ROOT_PATH, "testappend/")
print(DATA_DIR)

exp_name = "exp8"
model_path_idx = 1
model_path = "trained_weights/ourTestData/run_"+str(model_path_idx)+"/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

#Loading all the data
x_train = np.load(os.path.join(DATA_DIR, "images_7.npy"))
y_train = np.load(os.path.join(DATA_DIR, "masks_7.npy"))
nb_cases = x_train.shape[0]
ind_list = [i for i in range(nb_cases)]
shuffle(ind_list)
nb_valid = int(nb_cases*0.2)
x_valid, y_valid = x_train[ind_list[:nb_valid]], y_train[ind_list[:nb_valid]]
x_train, y_train = x_train[ind_list[nb_valid:]], y_train[ind_list[nb_valid:]]

# y_valid = y_valid[:,:,:,0]
# y_train = y_train[:,:,:,0]
#y_valid = np.expand_dims(y_valid, axis=-1)
#y_train = np.expand_dims(y_train, axis=-1)

print(">> Train data: {} | {} ~ {}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print(">> Train mask: {} | {} ~ {}\n".format(y_train.shape, np.min(y_train), np.max(y_train)))
print(">> Valid data: {} | {} ~ {}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
print(">> Valid mask: {} | {} ~ {}\n".format(y_valid.shape, np.min(y_valid), np.max(y_valid)))

x_test = np.load(os.path.join(DATA_DIR, "images_3.npy"))
y_test = np.load(os.path.join(DATA_DIR, "masks_3.npy"))

#y_test = np.expand_dims(y_test, axis=-1)

print(">> Test  data: {} | {} ~ {}".format(x_test.shape, np.min(x_test), np.max(x_test)))
print(">> Test  mask: {} | {} ~ {}\n".format(y_test.shape, np.min(y_test), np.max(y_test)))

model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose', activation="softmax", classes=4)
model.compile('Adam', loss="categorical_crossentropy", metrics=['accuracy', mean_iou, dice_coef])
#model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=30, 
                                               verbose=1,
                                               mode='min',
                                              )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, exp_name+".h5"),
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only=True, 
                                              mode='min',
                                             )
callbacks = [check_point, early_stopping]

model.fit(x_train, y_train, epochs=20, callbacks=callbacks, validation_data=(x_valid, y_valid))





