# Saliency and Observer Disagreement Prediction with a Center-Biased Deep Encoder-Decoder

This repository contains the code of the Visual Saliency Prediction project for the Neural Network and Deep Learning course, which aims to predict visual saliency in images using deep learning techniques. For more details, please refer to the project report.

## Team

- [Elena Dal Bò](https://github.com/elenadb-pro) (matr. 2201050)
- [Jacopo Lorigiola](https://github.com/MasterGiola) (matr. 2185062)
- [Lorenzo Nardin](https://github.com/nardib) (matr. 2196372)

## Running in Colab

The Colab notebook `project_notebook_colab.ipynb` can clone the current repository, download the dataset and run the training and evaluation scripts. The [SALICON](https://www.salicon.net/) dataset used for training and evaluation is downloaded from a public [Google Drive folder](https://drive.google.com/file/d/1r0KX4hsfgyhU8nqjCDllMMorByagzMxB/view?usp=sharing). The [MIT1003](http://saliency.mit.edu/datasets.html) dataset used for testing is downloaded automatically from the authors' website.

The `configs/config.yaml` file centralizes hyperparameters and dataset paths so the pipeline can be tuned without editing source code.

> ⚠️ In order to evaluate a model previously trained, you have to upload it in `checkpoints/sweep`, set `RUN_TRAINING=False` and run all the cells. 

## Running in Local

The project codes can also be runned in local by using the `project_notebook_local.ipynb`. The rules described for the Colab also holds in this case.

## Project structure

```
project_notebook_colab.ipynb
project_notebook_local.ipynb
configs/
└── config.yaml                     # configuration file
README.md                           
requirements.txt                    # required dependencies
report.pdf                          # detailed project report
src/
├── agreement_data.py               # agreement dataset and loader, extends the saliency inputs with observer-agreement targets
├── agreement_losses.py             # agreement-head loss functions, including heteroscedastic NLL and gating
├── agreement_model.py              # wrapper model that adds an observer-agreement head to the saliency network
├── agreement_targets.py            # offline computation and caching of MIT1003 agreement targets
├── data.py                         # dataset loading, preprocessing, augmentation
├── download_observers.py           # download and verify MIT1003 per-observer fixation data
├── download.py                     # dataset downloading
├── evaluate.py                     # evaluation
├── experiments.py                  # experiment orchestration for tier sweeps and diagnostics
├── losses.py                       # loss functions
├── metrics.py                      # metrics for evaluation
├── model.py                        # network architecture
├── observers.py                    # MIT1003 fixation parsing and density rendering utilities
├── setup_check.py                  # local environment and data sanity checks
├── train_agreement.py              # training loop for the agreement head with cross-validated metrics
├── train.py                        # training loop for the main saliency model
├── utils.py                        # shared helpers (seeding, checkpointing, config loading)
└── visualize.py                    # visualization of predictions and metrics
```
