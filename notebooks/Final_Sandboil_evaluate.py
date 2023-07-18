import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is
from lib.metrics import create_dir
from lib.evaluate import test_model

from lib.load_data import get_data
from lib.metrics import jaccard, tversky, dice_coef, dice_loss, bce_dice_loss, focal_tversky_loss, bce_dice_loss_new,tversky_loss,create_dir, mcc_loss, mcc_metric

import cv2
from skimage import io
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow_addons as tfa

import matplotlib.font_manager
import matplotlib.pyplot as plt

import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json
tf.config.experimental.set_visible_devices([], 'GPU')

print('TensorFlow version: {version}'.format(version=tf.__version__))
print('Keras version: {version}'.format(version=tf.keras.__version__))
print('Eager mode enabled: {mode}'.format(mode=tf.executing_eagerly()))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))


class PCALayer(tf.keras.layers.Layer):
    def __init__(self, n_components, **kwargs):
        super(PCALayer, self).__init__(**kwargs)
        self.n_components = n_components

    def build(self, input_shape):
        self.shape = input_shape
        self.input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.n_components), dtype="float32",
                                      initializer='glorot_uniform',
                                      trainable=False)

    def call(self, x):
        # Flatten the input tensor
        batch_size = tf.shape(x)[0]
        flattened = tf.reshape(x, [batch_size, -1, self.input_dim])
        
        # Compute the mean and subtract it from the input tensor
        mean = tf.reduce_mean(flattened, axis=1, keepdims=True)
        centered = flattened - mean
        

        # Compute the covariance matrix
        cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flattened)[1] - 1, tf.float32)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = tf.linalg.eigh(cov)

        # Sort the eigenvectors based on the eigenvalues
        idx = tf.argsort(eigenvalues, axis=-1, direction='DESCENDING')
        top_eigenvectors = tf.gather(eigenvectors, idx, batch_dims=1, axis=-1)
        top_eigenvectors = top_eigenvectors[:, :, :self.n_components]

        # Transpose the eigenvectors to match the input shape
        top_eigenvectors = tf.transpose(top_eigenvectors, perm=[0, 1, 2])
        
        # Project centered data onto top principal components
        projected = tf.matmul(centered, top_eigenvectors)

        # Reshape projected data and return as output
        output_shape = tf.concat([tf.shape(x)[:-1], [self.n_components]], axis=0)
        output = tf.reshape(projected, output_shape)
        return output



    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.n_components,)

    def get_config(self):
        config = super(PCALayer, self).get_config()
        config.update({'n_components': self.n_components})
        return config


def load_model(model_path, model_name, pca_layer=False):
    
    path = os.path.join(model_path, str(model_name), "best_model.h5")
    if pca_layer:
        with tf.keras.utils.CustomObjectScope({'jaccard':jaccard}, {'dice_coef':dice_coef}, {'tversky':'tversky'},
                                               {'bce_dice_loss_new': bce_dice_loss_new}, {'PCALayer': PCALayer}):
            model = tf.keras.models.load_model(path)  
        print(f'{model_name} is loaded')
        return model
        
    else:
        with tf.keras.utils.CustomObjectScope({'jaccard':jaccard}, {'dice_coef':dice_coef}, 
                                               {'bce_dice_loss_new': bce_dice_loss_new}):
            model = tf.keras.models.load_model(path)
        print(f'{model_name} is loaded')
        return model

def evaluate_model_for_dataset(X_test, Y_test, model, model_name, dataset_name, threshold, custom_layer=False):
    img_height = 512
    img_width = 512    
    
    result_folder_name = str(model_name) + "_" + str(dataset_name)
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    tf.keras.backend.clear_session()
    print(f'=========Loaded {model_name} for {dataset_name}===========')
    
    results = test_model(model, X_test, Y_test, result_folder_name, dataset_name, threshold)
    print("==========Evaluation Completed============")


def main():

    height = 512
    width = 512


    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
    model_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    model_dir = os.path.join(model_dir+ "/models/IEEE_models/")
    models = os.listdir(model_dir)

    sandboil_data = os.path.join(root_dir, "datasets", "test") 
    sandboil_images = sorted(next(os.walk(sandboil_data + "/images"))[2])
   

    model_names = [item for item in models if os.path.isdir(os.path.join(model_dir, item))]
    
    
    model_list = []
    for model_name in model_names:
        model = load_model(model_dir, model_name,pca_layer=True)
        model_list.append((model, model_name))

    
    X_test_levee, Y_test_levee = get_data(sandboil_images, sandboil_data, height, width, train=True)
    
    
    dataset_list = [(X_test_levee, Y_test_levee, 'IEEE_models_evaluation')]
    threshold=0.5

    for model, model_name in model_list:
        for X_test, Y_test, dataset_name in dataset_list:
            evaluate_model_for_dataset(X_test, Y_test, model, model_name, dataset_name, threshold)

if __name__ == '__main__':
    main()
















