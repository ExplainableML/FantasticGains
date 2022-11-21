### Setup up the environment.
First: `conda create -n repsssl python=3.9` and `conda activate repsssl`.

Next, install ffcv via 
```
conda install cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c conda-forge -c pytorch
```

Then, simply install solo-learn related packages using

```
pip3 install .[dali,umap,h5] --extra-index-url https://developer.download.nvidia.com/compute/redist
```

__Note:__ If the ffcv installation does not succeed, try for example a

(1) successive installation:
```
pip install opencv-python
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install pkg-config compilers libjpeg-turbo numba -c conda-forge
pip install fastargs terminaltables cupy-cuda113 torchmetrics
pip install -r requirements.txt
pip install ffcv
```


### Sample runs.
There are several example configs for quick access placed in `scripts/quick_tests`. 

To train a simple SimCLR model on the ImageNette320 dataset, simply run 
```
python main_pretrain.py --config-path scripts/quick_tests/ --config-name imagenet100_simclr.yaml ++wandb.enabled=False
```


To train a simple Classification model on ImageNet:
```
python main_pretrain.py --config-path scripts/quick_tests/ --config-name imagenet100_simple_supervised.yaml ++wandb.enabled=False
```
Here, we leverage the fact that the BaseMethod already includes a supervised CE classification training objective, that is simply overwritten by other objectives that inherit from it. The number of classes is derived automatically if the datasets follows the standard dataset structure (train/classes, val/classes), but can also be passed via `++data.num_classes=N`.

__NOTE:__ A simple linear probe is trained online jointly during training on the detached features to allow for fast computation of probing scores! In general, there is a strong relation between online and offline probing scores!


### Adding a new dataset.
Please ensure that the dataloader outputs an iterable of form `[sample_idx, images, targets]`.

## ImageNet-100
* To create imagenet-100, go to  `scripts/utils`, then run `python make_imagenet100.py <path_to_imagenet> <target_imagenet100_path>`.

To edit in `solo/args/dataset.py`:
* `SUPPORTED_DATASETS`
  
To edit in `solo/args/linear.py`:
* `N_CLASSES_PER_DATASET`
* `SUPPORTED_DATASETS`
  
To edit in `solo/args/dataset.py`:
* `SUPPORTED_DATASETS`

To edit in `solo/args/pretrain.py`:
* `N_CLASSES_PER_DATASET`
* `SUPPORTED_DATASETS`

To edit in `solo/data/h5_dataset.py`:
* Not needed?

To edit in `solo/data/pretrain_dataloader.py`: 
* `prepare_dataset()`
* `MEANS_N_STD`

To edit in `solo/data/classification_dataloader.py`:
* `prepare_datasets()`
* `prepare_transforms()`
  
To edit in `solo/data/dali_dataloader.py`:
* `MEANS_N_STD`
* `ClassificationDALIDataModule`


### Where is stuff stored and what is logged?
__WandB:__ By default, training metrics and model graphs are logged to W&B.

__Models:__ `trained_models/simclr/run_handle`, containing the `args.json` file and a respective checkpoint file.


### Where is the main training happening?


### Todos
* [X] Print validation results
* [ ] Load ffcv dataloader.
* [ ] Restructure ffcv dataloader to give idx, [img], target. Adjust `base.py` > `training/validation_step()`.
* [ ] How to resume Checkpointing.
* [ ] Load VISSL checkpoints.
* [ ] Store metrics & log output.
* [ ] Check W&B project and group name changes.
* [ ] "Failed to sample/serialize metric" error.
* [ ] Train supervised model & compare to ffcv.
* [ ] Incorporate ffcv variant.
* [ ] Loading and evaluating trained models.