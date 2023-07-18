import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is
from lib.metrics import create_dir
from lib.evaluate import test_model
from lib.load_data import get_data

import cv2
from skimage import io
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow_addons as tfa

import matplotlib.font_manager
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json

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
        #x = tf.linalg.normalize(x,axis=-1)
        #print(x.shape)
        # assumption is that the feature vector is normalized
        #x = tf.math.l2_normalize(x, axis=-1)
        batch_size = tf.shape(x)[0]
        flattened = tf.reshape(x, [batch_size, -1, self.input_dim])
        print(f'shape of flattened {flattened.shape}')
        
        # Compute the mean and subtract it from the input tensor
        mean = tf.reduce_mean(flattened, axis=1, keepdims=True)
        centered = flattened - mean
        print(f'shape of centered {centered.shape}')

        # Compute the covariance matrix
        cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flattened)[1] - 1, tf.float32)
        print(f'shape of covariance matrix {cov.shape}')
        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = tf.linalg.eigh(cov)

        print(f'shape of eigenvalues and eigenvectors {eigenvalues.shape} {eigenvectors.shape}')
        # Sort the eigenvectors based on the eigenvalues
        idx = tf.argsort(eigenvalues, axis=-1, direction='DESCENDING')
        top_eigenvectors = tf.gather(eigenvectors, idx, batch_dims=1, axis=-1)
        top_eigenvectors = top_eigenvectors[:, :, :self.n_components]

        print(f'shape of eigen vector {top_eigenvectors.shape}')

        # Transpose the eigenvectors to match the input shape
        top_eigenvectors = tf.transpose(top_eigenvectors, perm=[0, 1, 2])
        
        # Project centered data onto top principal components
        projected = tf.matmul(centered, top_eigenvectors)

        # Reshape projected data and return as output
        output_shape = tf.concat([tf.shape(x)[:-1], [self.n_components]], axis=0)
        #output = tf.reshape(projected, shape=(-1, *self.output_shape))
        output = tf.reshape(projected, output_shape)
        return output



    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.n_components,)

    def get_config(self):
        config = super(PCALayer, self).get_config()
        config.update({'n_components': self.n_components})
        return config



def load_model(model_name, custom_layer=False):
    result_folder_name = str(model_name)
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    model_path = os.path.join(root_dir, "models", "sandboilnet_type_models_for_comparison", str(model_name), "best_model.h5")
    tf.keras.backend.clear_session()
    
    if custom_layer:
        best_model = tf.keras.models.load_model(model_path, custom_objects={'PCALayer': PCALayer}, compile=False)
        
    else:
        print("here")
        best_model = tf.keras.models.load_model(model_path,compile=False)
    print(f'=========Loaded {model_name}===========')
    return best_model

def evaluate_model_for_dataset(X_test, Y_test, model, model_name, dataset_name, custom_layer=False):
    img_height = 256
    img_width = 256    
    
    result_folder_name = str(model_name) + "_" + str(dataset_name)
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    tf.keras.backend.clear_session()
    print(f'=========Loaded {model_name} for {dataset_name}===========')
    #saveResultsOnly(best_model, X_test, 4, result_folder_name)
    results = test_model(model, X_test, Y_test, result_folder_name)
    print("==========Evaluation Completed============")


def main():

    height = 256
    width = 256


    root_dir = os.path.normpath(os.getcwd() +os.sep + os.pardir + os.sep + os.pardir)
    path_test_levee_2 = os.path.join(root_dir,  "datasets", 'original_dataset' ,"test") # test_images/no_grassland | with_grassland | puddle_dataset/on_road \off_road
    test_images_levee_2 = sorted(next(os.walk(path_test_levee_2 + "/images"))[2])

    # path_test_levee = os.path.join(root_dir,  "datasets", 'train_test_for_exp3', 'test') #exp_loss_evaluation_data for loss and train_test_for_exp3
    # test_images_levee = sorted(next(os.walk(path_test_levee + "/images"))[2])

    
    model_names = ["Baseline_Normal_IEEE_bce_dice_loss_new", "SandBoilNet_CBAM_bce_dice_loss_new", "SandBoilNet_SE_bce_dice_loss_new", "best_Sandboil_dropout_new_augmented_data_bce_dice_loss_new", 
    "SandboilNet_Low_Dimension_PCA_bce_dice_loss_new", "SandboilNet_Dropout_Without_PCA_bce_dice_loss_new"]
    #model_names = ["Baseline_Normal_bce_dice_loss_new"]
    
    model_list = []


    for model_name in model_names:
        model = load_model(model_name, custom_layer=True)
        model_list.append((model, model_name))

    X_test, Y_test = get_data(test_images_levee_2, path_test_levee_2, height, width, train=True)
    dataset_list = [(X_test, Y_test, "levee")]


    for model, model_name in model_list:
        for X_test, Y_test, dataset_name in dataset_list:
            evaluate_model_for_dataset(X_test, Y_test, model, model_name, dataset_name)

if __name__ == '__main__':
    main()