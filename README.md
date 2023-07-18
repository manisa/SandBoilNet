# Paper Title: Deep Learning Approach for Accurate Segmentation of Sand Boils in Levee Systems
This repository contains codebase and links for datasets of our paper based on controlled transfer learning. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

You would need to install the following software before replicating this framework in your local or server machine.

```
Python version 3.7+
Aanaconda version 3+
TensorFlow version 2.12.0
Keras version 2.12.0

```

## Download and install code
- Retrieve the code
```
git clone https://github.com/manisa/SandBoilNet.git
cd SandBoilNet
```

- Create and activate the virtual environment with python dependendencies. 
```
conda create -n gpu-tf tensorflow-gpu
conda activate gpu-tf
pip install tensorflow==2.12.*

```

## Folder Structure
```
SandBoilNet/
	archs/
	lib/
	datasets/
        sandboil_augmented_5_8_23_6853/
        test/
	models/
        IEEE_models/
            Baseline_Conv_bce_dice_loss/
            Baseline_LeakyRI_bce_dice_loss/
            baseline_normal_bce_dice_loss/
            Baseline_ProposedAtt_bce_dice_loss/
            SandBoilNet/
            unet_bce_dice_loss/

```

## Download datasets
- [Original Training Data](https://cs.uno.edu/~mpanta1/SandBoilNet/datasets/train.zip) 
- [Original Test Data](https://cs.uno.edu/~mpanta1/SandBoilNet/datasets/test.zip) 
- [Augmented Train and Validation Data](https://cs.uno.edu/~mpanta1/SandBoilNet/datasets/sandboil_augmented_5_8_23_6853.zip)
- Unzip and copy dataset from the respecitve experiment into the folder **datasets** inside the root folder **SandBoilNet**.


## Download trained models
- [All IEEE Access Models](https://cs.uno.edu/~mpanta1/SandBoilNet/models/IEEE_models.zip)
- Unzip and copy models from respective experiment to **models** inside the root folder **SandBoilNet**.
- Your directory structure should look like this:

## Training
- To replicate the training procedure, follow following command line.
```
cd src
python train.py

```

## Authors
MANISHA PANTA, KENDALL N. NILES, JOE TOM, MD TAMJIDUL HOQUE, MAHDI ABDELGUERFI AND MAIK FALANAGIN

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
