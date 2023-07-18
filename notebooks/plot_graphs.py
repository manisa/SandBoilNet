import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is
from glob import glob
from lib.metrics import create_dir
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.figure(figsize=(32, 16 )) # *width, height
plt.rcParams.update({'font.size' : 58})

def plot_history_of_model(model_name, loss_function, model):
    result_folder_name = str(model_name)
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    graph_path_loss = os.path.join(root_dir, "graphs", "models_ok", "Baseline_Comparison_bce_dice")

    create_dir(graph_path_loss)
    
    csv_path = os.path.join(root_dir, "models_ok", "Baseline_Comparison_bce_dice",
                            str(model_name) + "_" + str(loss_function) , "metrics_" + str(model_name) + ".csv")
    history_df = pd.read_csv(csv_path)    
    history_to_plot = history_df.iloc[2:, :]
    history_to_plot = history_to_plot.reset_index(drop=True)
    history_to_plot.columns = history_df.columns

    # find the index of the minimum 'val_loss'
    min_val_loss_index = history_to_plot['val_loss'].idxmin()

    plt.figure(figsize=(32, 16)) # *width, height
    plt.rcParams.update({'font.size' : 58})
    sns_plot = sns.lineplot(data=history_to_plot[['loss', 'val_loss']], linewidth=8)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Graph for {model}")

    # mark the epoch with minimum 'val_loss' using a cross marker
    plt.scatter(min_val_loss_index, history_to_plot.loc[min_val_loss_index, 'val_loss'], color='red', marker='x', s=1000)

    # Set xticks
    plt.xticks(np.arange(0, len(history_to_plot['loss']), step=4))

    sns_plot.figure.savefig(graph_path_loss + '/' + str(model_name) + '_loss' + '.png')
    plt.close()


plot_history_of_model("Baseline_Att_512", "bce_dice_loss_new" , "Baseline-PPCSA-LeakyRI")
plot_history_of_model("Baseline_CBAM_512", "bce_dice_loss_new", "Baseline-CBAM-LeakyRI" )
plot_history_of_model("Baseline_Conv_512", "bce_dice_loss_new", "Baseline-Conv" )
plot_history_of_model("Baseline_LeakyRI_512", "bce_dice_loss_new", "Baseline-LeakyRI" )
plot_history_of_model("Baseline_Normal_23", "bce_dice_loss_new", "Baseline" )
plot_history_of_model("Baseline_SE_512", "bce_dice_loss_new", "Baseline-SE-LeakyRI" )

