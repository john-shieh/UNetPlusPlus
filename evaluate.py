#John Shieh

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

#saved model path
model_path_idx = 1
model_path = "trained_weights/ourdata/run_"+str(model_path_idx)+"/"
exp_name="exp1"

#load data
x_test = np.load(os.path.join(config.DATA_DIR, "images_3.npy"))
y_test = np.load(os.path.join(config.DATA_DIR, "masks_3.npy"))

print(">> Test  data: {} | {} ~ {}".format(x_test.shape, np.min(x_test), np.max(x_test)))
print(">> Test  mask: {} | {} ~ {}\n".format(y_test.shape, np.min(y_test), np.max(y_test)))

#load saved model and evalute
model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose', activation="softmax", classes=4)
model.load_weights(os.path.join(model_path, exp_name+".h5"))
model.compile('Adam', loss="categorical_crossentropy", metrics=['accuracy', mean_iou, dice_coef])
eva = model.evaluate(x_test, y_test, batch_size=config.batch_size, verbose=config.verbose)

print(eva)
print(">> Testing dataset mDice = {:.2f}%".format(eva[3]*100.0))

