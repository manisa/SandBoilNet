import os
import time
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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

import matplotlib.font_manager
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow.keras.backend as K
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

        # Creating an empty array of shape 1,512,512,3
        X = np.empty((1,512,512,3))
        # Read the image
        img = io.imread(i)
        # Resizing the image and converting them to array of type float64
        img = cv2.resize(img, (512,512))
        img = np.array(img, dtype=np.float64)
        
        # Standardising the image
        img /= 255.0
        # Converting the shape of image from 512,512,3 to 1,512,512,3
        X[0,] = img
        
        # Make prediction of mask
        for model_name, model in model_dict.items():
            start_time = time.time()
            prediction = model.predict(X, verbose=0)
            prediction = K.greater_equal(prediction, 0.6)
            #prediction = np.round(prediction,0)
            end_time = time.time()
            eval_time = end_time - start_time
            print(f'eval time for {model_name} is {eval_time}seconds')
            results_dict[model_name].append(prediction)
        
        image_id.append(os.path.join(image_path, filename))
        mask_id.append(os.path.join(mask_path, filename.split('.')[0] + '.png'))
        has_mask.append(1)
            
    return pd.DataFrame({'file_name': image_names, 'image_path': image_id, 
                         'mask_path': mask_id, 'has_mask': has_mask, **results_dict})

def iou_metric(y_true_in, y_pred_in):
    smooth = 1.0
    intersection = y_true_in.ravel() * y_pred_in.ravel()
    union = y_true_in.ravel() + y_pred_in.ravel() - intersection

    iou = ((np.sum(intersection) + smooth)/(np.sum(union) + smooth))
    return iou



#visualizing prediction
img_height = 512
img_width = 512
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize


def show_visualization(df_pred, filename, model_dict, num_images):
    count = 0
    num_models = len(model_dict)
    fig, axs = plt.subplots(num_images, num_models + 1, figsize=(6 * (num_models + 1), 6 * num_images)) # (width, height)
    W, H = 512, 512
    dim = (W,H)
    
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    filepath = os.path.join(root_dir, "results", filename)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.3)
    
    for count, row in enumerate(df_pred.itertuples()):
        if row.has_mask == 1 and count < num_images:
            img = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, dim)
            axs[count, 0].imshow(img)
            axs[count, 0].axis('off')
            axs[count, 0].set_title('Original Image ' + str(count + 1), fontsize=48, pad=6)

            # mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
            # mask = cv2.resize(mask, dim)
            # img[mask == 255] = (200, 0, 0)
            # axs[count, 1].imshow(img)
            # axs[count, 1].axis('off')
            # axs[count, 1].set_title('Ground Truth ' + str(count + 1), fontsize=68, pad=6)
            # mask = np.array(mask, dtype=np.float64)
            # mask = mask / 255

            for j, (model_name, model) in enumerate(model_dict.items()):
                pred = np.array(row._asdict()[model_name])
                #iou = iou_metric(mask, pred)
                pred = np.squeeze(pred)
                pred = np.uint8(pred > 0.5)
                img_model = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
                img_model = cv2.resize(img_model, (512, 512))
                img_model = img_model.squeeze()
                #img_model[pred == 1] = [(255, 255, 0), (204, 0, 204), (51, 51, 255)][j]
                img_model[pred == 1] = [ (0, 255, 0), (0, 255, 255), (127, 0, 255), (100, 105, 204), (204, 0, 204)][j] #(255, 255 , 255), 


                # Find contours of the segmented regions
                #contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw bounding boxes around the segmented regions
                #for contour in contours:
                 #   x, y, w, h = cv2.boundingRect(contour)
                  #  cv2.rectangle(img_model, (x, y), (x+w, y+h), (0, 210, 0), 4)
                    #cv2.putText(img_model, f"IoU: {iou:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    
                axs[count, j + 1].imshow(img_model)
                axs[count, j + 1].axis('off')
                axs[count, j + 1].set_title(f'[M{j + 1}]', fontsize=48, pad=6)

    fig.tight_layout()
    plt.savefig(filepath, format='png', facecolor="w", transparent=False)
    plt.close()


def load_model(model_name, custom_layer=False):
    result_folder_name = str(model_name)
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    model_path = os.path.join(root_dir,  "models/IEEE_Access_Models_trained_on_6853images/state_of_the_art_models_4e_4", str(model_name), "best_model.h5") #state_of_the_art_models_4e_4 sandboilnet_variations_4e_4
    tf.keras.backend.clear_session()
    
    if custom_layer:
        best_model = tf.keras.models.load_model(model_path, custom_objects={'PCALayer': PCALayer}, compile=False)
        
    else:
        print("here")
        best_model = tf.keras.models.load_model(model_path,compile=False)
    print(f'=========Loaded {model_name}===========')
    return best_model

## Load data for visualization

img_height = 512
img_width = 512
import random
root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir )
print(root_dir)

path_test_levee = os.path.join(root_dir,  "datasets","test_images", "levee_crack_test") #crack_negative_test burrows_test pothole_test seepage_test exp2_test levee_crack_test
print(path_test_levee)

test_images_leveecrack = sorted(next(os.walk(path_test_levee + "/images"))[2])

#random.Random(42).shuffle(test_images_leveecrack)



print(test_images_leveecrack)
image_leveecrack_path = path_test_levee + "/images"
mask_leveecrack_path = path_test_levee + "/masks"

M0 = load_model( 'unet_bce_dice_loss_new', False)
M1 = load_model('multiresunet_bce_dice_loss_new', False)
M2 = load_model('attentionunet_bce_dice_loss_new', False)
M3 = load_model('nestedunet_bce_dice_loss_new', False)
M4 = load_model('SandBoilNet_Low_Dimension_PCA_bce_dice_loss_new', True)


# M0 = load_model('baseline_normal_bce_dice_loss_new', False)
# M1 = load_model('SandBoilNet_CBAM_bce_dice_loss_new', True)
# M2 = load_model('SandBoilNet_SE_bce_dice_loss_new', True)
# M3 = load_model('SandBoilNet_Dropout_Without_PCA_bce_dice_loss_new', False)
# M4 = load_model('SandBoilNet_4e_4_noDropout_bce_dice_loss_new', True)
# M5 = load_model('SandBoilNet_Low_Dimension_PCA_bce_dice_loss_new', True)

#model_dict = {'M1':M0, 'M2':M1, 'M3':M2, 'M4':M3, 'M5':M4, 'M6':M5}


model_dict = {'M1':M0, 'M2':M1, 'M3':M2, 'M4':M3, 'M5':M4}

df_all_preds_all = []
num_images = 4

df_all_preds_all = prediction(test_images_leveecrack, os.path.abspath(image_leveecrack_path), os.path.abspath(mask_leveecrack_path), model_dict)
#df_for_visualization = df_all_preds_all.sample(frac = 0.98)
show_visualization(df_all_preds_all, "levee_crack_test.png", model_dict, num_images)