import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, Concatenate, DepthwiseConv2D, Dense, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate, Reshape, multiply, add, Permute, Add, Concatenate
from tensorflow.keras.models import Model, model_from_json



def initial_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='gelu', name=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters,(num_row, num_col),padding='same',kernel_regularizer=None, kernel_initializer=tf.keras.initializers.HeNormal(seed=2023))(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Activation(activation, name=name)(x)
    return x



def conv2d_bn(x, filters, num_row, num_col):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    x = initial_conv2d_bn(x, filters, num_row, num_col)
    return x

def decoder_block(inputs, skip, filters):
    x = tf.keras.layers.UpSampling2D((2, 2),interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip])
    x = conv2d_bn(x, filters, 3, 3)
    return x


def Baseline_Conv(input_filters, height, width, n_channels):
    #inputs = Input((height, width, n_channels), name = "input_image")
    filters = input_filters
    model_input = Input(shape=(height, width, n_channels))
    
    """ Pretrained resnet"""
    tf.keras.backend.clear_session()
    
    base_model = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_tensor=model_input, pooling=max)

    print("Number of layers in the base model: ", len(base_model.layers))
 

    base_model.trainable = True
    
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    for i, layer in enumerate(base_model.layers[:-48]):
        layer.trainable = False

    """ Encoder """
    s11 = (base_model.get_layer("input_1").output)
    s21 = (base_model.get_layer("conv1_conv").output)    
    s31 = (base_model.get_layer("conv2_block3_1_conv").output)    
    s41 = (base_model.get_layer("conv3_block4_1_conv").output)    


    """ Bridge """
    b11 = (base_model.get_layer("conv4_block6_1_conv").output)   
    b11 = conv2d_bn(b11, filters*8, 3, 3)
    

    """ Decoder """
    d11 = decoder_block(b11, s41, filters*4)                      
    d21 = decoder_block(d11, s31, filters*2)                        
    d31 = decoder_block(d21, s21, filters*1)                        
    d41 = decoder_block(d31, s11, filters//2)                       

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name='visualized_layer')(d41)
    model = Model(model_input, outputs, name="Baseline_Conv")

    return model



def main():

# Define the model

    model = Baseline_Conv(32, 512, 512, 3)
    print(model.summary())



if __name__ == '__main__':
    main()