# Exercise 3: Land Segmentation using UNET

(For nicer reading in Jupyter, righ-click and select `Show Markdown Preview`.) 

In this exercise, the land use classes are predicted with a UNET model using Pytorch and Torchgeo libraries. For comparison also a simple ResNet is trained. 

The data used for the exercise is pre-loaded in the data folder. In case you are interested, you can take a look at the raster data preparation notebook [raster_data_preparation.ipynb](raster_data_preparation.ipynb).

Satellite images are usually too big for CNN models as such, se we need to tile them to smaller tiles for training the model and also later for prediction. Torchgeo has very nice functionality for tiling and sampling the data for training. Unfortunatelly similar functionality does not exist for inference.

This exercise includes two steps:
* Model training, including data loading and tiling with torchgeo. This part is run as batch job, because GPU-resources are needed.
* Inference and evaluation of the model visually and by calculating performance metrics.

## Model training as a batch job.
* Open these files, we will go through it in details.
    * Python file with PyTorch code: [train_model.py](train_model.py)
    * HPC batch job file: [train_model.sh](train_model.sh)
* Change to the code whether you want to train a ResNet or the UNET
* Submit Python script as SLURM batch job in a supercomputer:
    * Open Terminal to login-node: Open Apps -> Login node shell
    * A black window with SSH connection to Mahti opens, now Linux commands should be used.
    * The shell opens in home directory, to access the files, change working 
    directory:
        * `cd /scratch/project_2017263/$USER/lumi-aif-fmi/day2/exercise3`
    * See that you are in the right folder:
        * `ls -l`.
        * It should list the files that you see also in Jupyter File panel.
    * Submit a batch job:
        * `sbatch train_model.sh`
    * It outputs the job number, for example: `Submitted batch job 1212121212`
* To see the Python output file, open it with `tail`, the exact file name depends on the number printed previosly:
    * `tail -f slurm-1212121212.out`.
    * The output file includes:
        * Printout of used folders, just to double-check
        * Results of each epoch. 
        * This output file is also the first place to look for errors, when writing own scripts.
    * Optional, to see full output from beginning:
        * `less slurm-1212121212.out`
        * This does not update, if file gets more rows.
    * It is possible to see job's state (waiting, running, finished) and used resources with
        * `sacct -o jobid,partition,state,reqmem,maxrss,averss,elapsed`
        * (In CSC Mahti: `seff 1212121212`)
* There should be new files in the `model_training` folder:
    *  The trained model as a .pt file
    *  Visualisation of the training and validation loss

## Inference and evaluation of the model visually and by calculating performance metrics.
* Open Jupyter as described in [main Readme](../../README.md)
* Open [inference_and_evaluation.ipynb](inference_and_evaluation.ipynb)
* Results are stored in 'inference' folder
