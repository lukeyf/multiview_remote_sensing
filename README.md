# Uncertainty-Aware Regression for Socio-Economic Estimation via  Multi-View Remote Sensing

## Introduction

This is a repository for the work **Uncertainty-Aware Regression for Socio-Economic Estimation via Multi-View Remote Sensing**.


## Getting DHS Data and Satellite Imagery


The work is developed based on [KidSat](https://github.com/MLGlobalHealth/KidSat), and we followed the procedures listed in this repository to obtain the necessary data for analysis. As the DHS survey data is sensitive and requires authorized access, we will not host the dataset in this repository. To replicate the experiments, please follow the instructions in the KidSat repository for obtaining the dataset.

At the end of the data preparation and pre-processing, you should have a directory containing country-wise imagery and a train-test splits in `survey_processing/processed_data`.

## Experiments
The follow instruction is similar to what is presented in the KidSat repository. 
## Experiment with DINOv2

The  `survey_processing/processed_data`, you can finetune DINOv2 using the following commands. For the spatial experiment with Landsat imagery, you can use the following code.

Finetuning sentinel imagery, the normal command is 

```bash
python modelling/dino/finetune_spatial.py --fold 1 --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 1 --imagery_source S --num_epochs 20 --grouped_bands 4 3 2
```

Note that to get a cross-validated result, you should use fold 1 to 5. For each view, you should modify the `--grouped_bands` argument. For the paper, we used these 4 views `[[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]]`. The other combinations could also be leveraged if deemed useful.

For evaluation, make sure the all 1-5 finetuned spatial models  (or the finetuned temporal model for temporal evaluation) are in `modelling/dino/model` and run 

```bash
python modelling/dino/evaluate.py --use_checkpoint --imagery_path {path_to_parent_imagery_folder} --imagery_source S --mode spatial --grouped_bands 4 3 2
```

We include an example script for the final regression [here](modelling/dino/results.ipynb). After all views are fine-tuned you can follow this notebook to evaluate the multi-view model.

## Experiment with SatMAE


### Finetuning
To run the finetuning process, you first need to download the checkpoint for fMoW-SatMAE [non-temporal](https://zenodo.org/record/7369797/files/fmow_pretrain.pth). Then run the following:

```sh
python -m modelling.satmae.satmae_ms_finetune --pretrained_ckpt $CHECKPOINT_PATH --dhs_path ./survey_processing/processed_data/train_fold_1.csv --output_path $OUTPUT_DIR --imagery_path $IMAGERY_PATH
```
Arguments:
- `--pretrained_ckpt`: Checkpoint of pretrained SatMAE model.
- `--imagery_path`: Path to imagery folder
- `--dhs_path`: Path to DHS `.csv` file
- `--output_path`: Path to export the output. A unique subdirectory will be created.
- `--batch_size`
- `--random_seed`
- `--sentinel`: Landsat is used by default. Turn this on to use Sentinel imagery
- `--temporal`: Add this flag to use the temporal mode
- `--epochs`: Number of epochs
- `--stopping_delta`: Delta for early stopping
- `--stopping_patience`: Early stopping patience
- `--loss`: Either `l1` (default) or `l2`.
- `--lr`: Learning rate
- `--weight_decay`: Weight decay for Adam optimizer
- `--enable_profiling`: Enable reporting of loading/inference time.


### Evaluation
Evaluation consists of 2 steps: exporting the model output, and perform Ridge Regression. Since exporting the model output is expensive, we split it into 2 separate modules:

To carry out the first step, edit the file `modelling/satmae/satmae_ms_eval` and change the `SATMAE_PATHS` variable accordingly. For each entry, you can put all the model checkpoints you need to evaluate or `None` to use the pretrained checkpoint, along with their fold (1-5). You do not have to put the entries in any order, nor need to put all the folds, but the script caches the data from different folds in memory, which helps significantly reduce the time for loading and preprocessing the satellite images.
```sh
python -m modelling.satmae.satmae_ms_eval --output_path $OUTPUT_DIR --imagery_path $IMAGERY_PATH
```
Arguments
- `--imagery_path`: Path to imagery folder
- `--output_path`: Path to export the output. A unique subdirectory will be created.
- `--batch_size`
- `--sentinel`: Landsat is used by default. Turn this on to use Sentinel imagery
- `--temporal`: Add this flag to use the temporal mode

This will export data as Numpy arrays in `.npy` files in the output location, which has the shape `(num_samples, 1025)`. The first 1024 columns (i.e `arr[:, :1024]`) is the predicted feature vector from the model, and the last column (i.e `arr[:, 1024]`) is the target. 
