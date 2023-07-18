import os
from pathlib import Path
from csv import writer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# it generates dicecoefficient over entire batch because of axis=0
def dice_coefficient(y_true, y_pred):
    smooth = 1e-7
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f, axis=0)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f, axis=0) + K.sum(y_pred_f, axis=0) + smooth)
    
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def jaccard(y_true, y_pred):
    def f(y_true, y_pred):
        smooth = 1e-15
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Crossentropy Loss and Dice Loss function for semantic segmentation.
    :param y_true: ground truth mask.
    :param y_pred: predicted mask.
    :return: the combined loss.
    """
    # Binary Crossentropy Loss
    epsilon = K.epsilon()
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Dice Loss
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred) 
    dice_loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth) 
    
    # Combine the two losses with equal weights
    combined_loss = (0.5 * bce) + (0.5 * dice_loss)
    
    return combined_loss