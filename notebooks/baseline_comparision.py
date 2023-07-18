import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is

from lib.metrics import create_dir
from lib.evaluate import testModel
from lib.load_data import get_data

import cv2
from skimage import io
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import matplotlib.font_manager
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json

print('TensorFlow version: {version}'.format(version=tf.__version__))
print('Keras version: {version}'.format(version=tf.keras.__version__))
print('Eager mode enabled: {mode}'.format(mode=tf.executing_eagerly()))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

def prediction(test_ids, image_path, mask_path, model_dict):
    # empty list to store results
    image_names, image_id, mask_id, has_mask = [], [], [], []

    results_dict = {name: [] for name in model_dict.keys()}
    
    image_path = os.path.abspath(image_path)
    mask_path = os.path.abspath(mask_path)

    # Iterating through each image in test data
    for n, i in tqdm(enumerate(test_ids), total=len(test_ids)):
        filename = i
        image_names.append(filename)
        i = os.path.join(image_path, i)

        # Creating an empty array of shape 1,256,256,3
        X = np.empty((1,256,256,3))
        # Read the image
        img = io.imread(i)
        # Resizing the image and converting them to array of type float64
        img = cv2.resize(img, (256,256))
        img = np.array(img, dtype=np.float64)
        
        # Standardising the image
        img /= 255.0
        # Converting the shape of image from 256,256,3 to 1,256,256,3
        X[0,] = img
        
        # Make prediction of mask
        for model_name, model in model_dict.items():
            prediction = model.predict(X)
            results_dict[model_name].append(prediction)
        
        image_id.append(os.path.join(image_path, filename))
        mask_id.append(os.path.join(mask_path, filename.split('.')[0] + '.png'))
        has_mask.append(1)
            
    return pd.DataFrame({'file_name': image_names, 'image_path': image_id, 
                         'mask_path': mask_id, 'has_mask': has_mask, **results_dict})

def iou_metric(y_true_in, y_pred_in):
    smooth = 1e-15
    intersection = y_true_in.ravel() * y_pred_in.ravel()
    union = y_true_in.ravel() + y_pred_in.ravel() - intersection

    iou = ((np.sum(intersection) + smooth)/(np.sum(union) + smooth))
    return iou

#visualizing prediction
img_height = 256
img_width = 256
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize


def show_visualization(df_pred, filename, model_dict):
    count = 0
    fig, axs = plt.subplots(10,7, figsize=(64, 84)) # (width, height)
    W, H = 256, 256
    dim = (W,H)
    
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    filepath = os.path.join(root_dir, "results", filename)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.3)
    
    for i in range(len(df_pred)):
        if df_pred.has_mask[i]==1 and count<10:
            
            #read levee sand boil image
            img = cv2.imread(df_pred.image_path[i], cv2.IMREAD_COLOR)
            img = cv2.resize(img, dim)
            axs[count][0].imshow(img)
            axs[count][0].axis('off') 
            axs[count][0].set_title('Original Image ' + str(count+1), fontsize=78, pad=6)

            #read original mask and overlay original mask with levee crack image    
            mask = cv2.imread(df_pred.mask_path[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, dim)
            img[mask==255] = (255,51,51)
            axs[count][1].imshow(img)
            axs[count][1].axis('off')
            axs[count][1].set_title('Ground Truth '+ str(count+1), fontsize=78, pad=6)
            mask = np.array(mask, dtype = np.float64)
            mask = mask / 255
            
            # Plotting for different models
            
            for j, model_name in enumerate(model_dict.keys()):
                pred = np.array(df_pred[model_name][i])
                iou = iou_metric(mask, pred )
                pred = np.round(pred,0).squeeze()
                img_model = cv2.imread(df_pred.image_path[i], cv2.IMREAD_COLOR)
                img_model = cv2.resize(img_model, (256, 256))
                img_model = img_model.squeeze()
                img_model[pred==1] = [(153, 153, 0), (51, 102, 0), (0, 153, 76), (0, 153, 153), (204, 0, 204)][j]
                axs[count][j+2].imshow(img_model)
                axs[count][j+2].axis('off')
                axs[count][j+2].set_title(f'[IoU IM{j}={iou:.2f}]', fontsize=78, pad=6)
            
            fig.tight_layout()
            plt.savefig(filepath, format='png', facecolor="w", transparent=False)
            
            count +=1
        if (count==10):
            break
    fig.tight_layout() 
    plt.close()
## Load data for visualization

img_height = 256
img_width = 256
import random
root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
print(root_dir)
#path_test_levee = os.path.join(root_dir,  "datasets", 'original_dataset' ,"test")
path_test_levee = os.path.join(root_dir,  "datasets" ,"original_dataset" , "test") #with_grassland, #no_grassland inside "test_images"
print(path_test_levee)

test_images_leveecrack = sorted(next(os.walk(path_test_levee + "/images"))[2])

random.Random(6464).shuffle(test_images_leveecrack)



print(test_images_leveecrack)
image_leveecrack_path = path_test_levee + "/images"
mask_leveecrack_path = path_test_levee + "/masks"


## Load Models

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
        #output = tf.reshape(projected, shape=(-1, *self.output_shape))
        output = tf.reshape(projected, output_shape)
        return output



    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.n_components,)

    def get_config(self):
        config = super(PCALayer, self).get_config()
        config.update({'n_components': self.n_components})
        return config


def evaluate_model_for_dataset(X_test, Y_test, model_name, dataset_name, custom_layer=False):
    img_height = 256
    img_width = 256    
    
    result_folder_name = str(model_name) + "_" + str(dataset_name)
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    model_path = os.path.join(root_dir,  "models", "final_baseline_models_for_comparison", str(model_name), "best_model.h5")
    tf.keras.backend.clear_session()
    if custom_layer:
        best_model = tf.keras.models.load_model(model_path, custom_objects={'PCALayer': PCALayer}, compile=False)
    
    else:
        print("here")
        best_model = tf.keras.models.load_model(model_path,compile=False)
        
    print(f'=========Loaded {model_name} for {dataset_name}===========')
    #saveResultsOnly(best_model, X_test, 4, result_folder_name)
    results = testModel(best_model, X_test, Y_test, 4, result_folder_name)
    print("==========Evaluation Completed============")

def load_model(model_name, custom_layer=False):
    result_folder_name = str(model_name)
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    model_path = os.path.join(root_dir,  "models", "final_baseline_models_for_comparison", str(model_name), "best_model.h5")
    tf.keras.backend.clear_session()
    
    if custom_layer:
        best_model = tf.keras.models.load_model(model_path, custom_objects={'PCALayer': PCALayer}, compile=False)
        
    else:
        print("here")
        best_model = tf.keras.models.load_model(model_path,compile=False)
    print(f'=========Loaded {model_name}===========')
    return best_model
#for_journal/models/better_baseline_models_with_conv_block


M00 = load_model("Baseline_Conv_IEEE_bce_dice_loss_new", custom_layer=False)
M0 = load_model("Baseline_LeakyRI_IEEE_bce_dice_loss_new", custom_layer=False)
M1 = load_model("Baseline_CBAM_IEEE_bce_dice_loss_new", custom_layer=False)
M2 = load_model("Baseline_SE_IEEE_bce_dice_loss_new", custom_layer=False)
M3 = load_model("Baseline_Att_IEEE_bce_dice_loss_new", custom_layer=False)


# model_dict = {'Baseline_conv_8_bce_dice_loss_new':M00, 'Baseline_iterblock_8_bce_dice_loss_new':M0, 'iterlblock_CBAM_8_bce_dice_loss_new': M1, 'iterlblock_SE_8_bce_dice_loss_new': M2, 
#               'iterlblock_att_8_bce_dice_loss_new': M3}



model_dict = {'IM0':M00, 'IM1':M0, 'IM2':M1, 'IM3':M2, 'IM4':M3}

df_all_preds_all = []


df_all_preds_all = prediction(test_images_leveecrack, os.path.abspath(image_leveecrack_path), os.path.abspath(mask_leveecrack_path), model_dict)
df_for_visualization = df_all_preds_all.sample(frac = 1)
show_visualization(df_for_visualization, "baseline_comparision.png", model_dict)