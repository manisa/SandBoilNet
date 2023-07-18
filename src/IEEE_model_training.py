import os
import time
import argparse
import random
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import tensorflow_addons as tfa


import sys; sys.path.insert(0, '..')
from lib.metrics import jaccard, dice_coef, dice_loss, bce_dice_loss, create_dir
from lib.dataloader import SandboilDataGen

from archs.SandBoilNet import SandboilNet
from archs.baseline_conv_model import Baseline_Conv
from archs.baseline_leakyRI import Baseline_LeakyRI
from archs.baseline_model import Baseline_Normal
from archs.baseline_proposed_att import Baseline_ProposedAtt


print('TensorFlow version: {version}'.format(version=tf.__version__))
print('Eager mode enabled: {mode}'.format(mode=tf.executing_eagerly()))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

def save_text_to_file(filepath, line, model_name):
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a') as file:
        if file_exists:
            file.write('\n')  # Add a newline before appending new lines
            file.write('Training time for ' + str(model_name) + ' = ' + str(line) + ' Hours ' + '\n')


def compile_and_train_model(config, valid_ids, train_test_ids):
    
    filename = "models_train_time.txt"
    save_train_time = os.path.join(config.all_models_path, filename)
    print(save_train_time)

    print('Loading dataset...')
    
    train_ids = train_test_ids
    for arch in config.model_list:
        tf.keras.backend.clear_session()
        config.model_type = arch
        print(config.model_type)
        print('Creating directories to store model and results...')
        
        model_path =  config.model_type + "_" + config.loss_function 
        config.model_path = os.path.join(config.all_models_path, model_path)
        create_dir(config.model_path)
        print(config)



        if config.model_type == "SandboilNet":
            model = SandboilNet(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

        elif config.model_type == "Baseline_Conv":
            model = Baseline_Conv(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

        elif config.model_type == "Baseline_LeakyRI":
            model = Baseline_LeakyRI(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

        elif config.model_type == "Baseline_Normal":
            model = Baseline_Normal(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)
        
        elif config.model_type == "Baseline_ProposedAtt":
            model = Baseline_ProposedAtt(input_filters=config.input_filters*2, height=config.img_height, width=config.img_width, n_channels=config.img_ch)
        

        

        # Defining optimizer
        if config.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, beta_1=config.beta1, beta_2=config.beta2)
        elif config.optimizer == "Lazy_Adam":
            optimizer = tfa.optimizers.LazyAdam(learning_rate=config.lr, beta_1=config.beta1, beta_2=config.beta2)
        elif config.optimizer == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=config.lr, beta_1=config.beta1, beta_2=config.beta2)
        elif config.optimizer == "RMSProp":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.lr, momentum=config.beta1)
        elif config.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.lr, momentum=config.beta1)


            # Defining loss functions
        if config.loss_function == "bce":
            model_loss = tf.keras.losses.BinaryCrossentropy(
                                            from_logits=True,   
                                            label_smoothing=0.0,
                                            axis=-1,
                                            reduction="auto",
                                            name="binary_crossentropy")
        elif config.loss_function == "dice_loss":
            model_loss = dice_loss
        elif config.loss_function == "bce_dice_loss":
            model_loss = bce_dice_loss
        


        # define metrics
        metrics =[dice_coef, jaccard, 'accuracy']
        model.compile(loss=model_loss, optimizer=optimizer, metrics=metrics)
        print(f'model created and compiled for model {config.model_type}')
        print(model.summary())


        csv_path = config.model_path + "/metrics_" + config.model_type + ".csv"
        
        print("Loading data generator...")
        image_size = (config.img_height, config.img_width)
        train_generator = SandboilDataGen(train_ids, config.train_valid_path, img_height=config.img_height, img_width=config.img_width, batch_size=config.batch_size)
        valid_generator = SandboilDataGen(valid_ids, config.train_valid_path, img_height=config.img_height, img_width=config.img_width, batch_size=config.batch_size)


        steps_per_epoch = len(train_ids) // config.batch_size
        val_steps_per_epoch = len(valid_ids) // config.batch_size

        print(f' Training and Validation steps are {steps_per_epoch}, {val_steps_per_epoch}')
        
        
        
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.06, patience=6, verbose=1, mode='min', min_delta=0.001, cooldown=0, min_lr=0.00000000000000001),
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1),
                            tf.keras.callbacks.CSVLogger(csv_path),
                            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config.model_path, 'best_model.h5'), monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=False, verbose=1)]
                            

        print(f'Fitting model {config.model_type}...')


        start_time = time.time()
        history = model.fit(train_generator, validation_data=valid_generator,
                                        steps_per_epoch=steps_per_epoch, 
                                        validation_steps=val_steps_per_epoch,
                                        epochs=config.num_epochs, callbacks=callbacks, shuffle=True,  verbose=2)
        end_time = time.time()
        training_time = (end_time - start_time) / 3600

        print(f'Training time for {config.model_type} is {training_time} hours')
        save_text_to_file(save_train_time, training_time, config.model_type)


        
        tf.keras.backend.clear_session()
        
        print(f'==================Model {config.model_type} training completed====================')
    


def main(config):
    # create generalied directories

    create_dir(config.all_models_path)
    create_dir(config.all_results_path)
    
    image_filenames = sorted(glob(os.path.join(config.train_valid_path, "images/*")))
    random.Random(config.seed).shuffle(image_filenames)

    
    
    # Define number of train and validation images
    total_samples= len(image_filenames)
    NUM_VAL_IMAGES = int(total_samples * config.valid_perc)
    NUM_TRAIN_IMAGES = total_samples - NUM_VAL_IMAGES 

    # Select train and valid ids
    valid_ids = image_filenames[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]
    train_ids = image_filenames[:NUM_TRAIN_IMAGES]
    print(len(valid_ids), len(train_ids))

    compile_and_train_model(config, valid_ids, train_ids)


if __name__ == '__main__': 
    # model hyper-parameters

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--input_filters', type=int, default=32)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4e-4) 
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_function', type=str, default='bce_dice_loss')
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)      # momentum2 in Adam    

    parser.add_argument('--model_list', type=str, default=['SandboilNet', 'Baseline_Conv', 'Baseline_LeakyRI', 'Baseline_Normal', 'Baseline_ProposedAtt'])
    parser.add_argument('--all_models_path', type=str, default='../models/')
    parser.add_argument('--all_results_path', type=str, default='../results/')

    parser.add_argument('--train_valid_path', type=str, default='../datasets/sandboil_augmented_5_8_23_6853/')
    parser.add_argument('--test_path', type=str, default='../../datasets/test/')
    parser.add_argument('--valid_perc', type=float, default=0.26)
    parser.add_argument('--test_perc', type=float, default=0.0)
    

    parser.add_argument('--seed', type=int, default=2023)
    config = parser.parse_args()
    main(config)