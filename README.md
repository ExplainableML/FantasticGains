# Fantastic Gains and Where to Find Them: On the Existence and Prospect of General Knowledge Transfer between Any Pretrained Model

This repository contains the codebase for the paper "[Fantastic Gains and Where to Find Them: On the Existence and Prospect of General Knowledge Transfer between Any Pretrained Model](https://arxiv.org/abs/2310.17653)" published at ICLR 2024 as a spotlight paper.

![Figure 3](images/MT-Dist-Figure7.png)

## Introduction
Training deep networks requires various design decisions regarding for instance
their architecture, data augmentation, or optimization. In this work, we find these
training variations to result in networks learning **unique** feature sets from the data.
Using public model libraries comprising thousands of models trained on canonical
datasets like ImageNet, we observe that for arbitrary pairings of pretrained models,
one model extracts significant data context unavailable in the other – independent
of overall performance. Given **any arbitrary pairing of pretrained models** and no
external rankings (such as separate test sets, e.g. due to data privacy), we investigate
if it is possible to transfer such "complementary" knowledge from one model to
another without performance degradation – a task made particularly difficult as
additional knowledge can be contained in stronger, equiperformant or weaker
models. Yet facilitating robust transfer in scenarios agnostic to pretrained model
pairings would unlock **training guidance, auxiliary gains and knowledge fusion**
from any model repository without restrictions on model & problem specifics – including from **weaker, lower-performance models**. This work provides a
first, in-depth exploration on the viability of such **general-purpose knowledge
transfer**. Across large-scale experiments, we first reveal the shortcomings of
standard knowledge distillation techniques, and then propose a general extension
via data partitioning for successful transfer between nearly all pretrained models -
which can also be done **unsupervised**. Finally, we assess both the scalability and
impact of model properties on successful model-agnostic knowledge transfer.

## Features
 - **Complementary Knowledge Discovery**: Demonstrates the consistent existence of complementary knowledge between any pair of pretrained models, regardless of their performance rankings or architecture types.
 - **Knowledge Transfer Methodology**: Introduces a novel data partitioning method for general-purpose knowledge transfer, enabling efficient transfer between pretrained models without performance degradation.
 - **Continual Learning Approach**: Treats the transfer process as a continual learning problem to balance knowledge gain and retention, ensuring robust performance.
 - **Unsupervised Transfer**: Proposes unsupervised methods for knowledge transfer, eliminating the need for labeled data during the transfer process.
 - **Multi-Teacher Knowledge Transfer**: Explores the feasibility of transferring knowledge from multiple pretrained models, leveraging various strategies like parallel and sequential transfers. 
 - **Scalability and Generalization**: Validates the scalability of the proposed methods and their ability to generalize across different datasets and model architectures. 
 - **Extensive Experiments**: Conducts large-scale experiments to demonstrate the effectiveness of the proposed methods, highlighting significant improvements over traditional knowledge distillation techniques.

## Setup
To get started with the codebase, follow these steps:
1. Cloning the repository.
    ```sh
    git clone https://github.com/username/repository.git
    cd repository
    ```
2. Setting up the environment (install ffcv following the [official documentation](https://ffcv.io/)).
    ```sh
    conda create -n ffcv python=3.9 cupy pkg-config libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.6 numba -c conda-forge -c pytorch 
    conda activate ffcv 
    conda update ffmpeg 
    pip install ffcv        
    pip install -r requirements.txt
    ```

## Usage
Please find the instructions for replicating the experiments in the paper below. This codebase uses [hydra](https://hydra.cc/docs/intro/) for configuration management. The config files of the experiments can be found in the `config` directory. Make sure to set the correct data paths in the config files before running the experiments.
Experimental results are logged using [wandb](https://docs.wandb.ai/guides). The wandb configuration can be found in the `config/wandb` directory.

**Complementary knowledge discovery** - The experiments for the complementary knowledge discovery can be run using the following command:
   ```python
    python main_flips.py --config-path config --config-name prediction_flips/flips.yaml
   ```

**Pretraining** - Experiments on datasets other than imagenet require pretraining all student and teacher models. For pretraining please execute the following command:
   ```python
    python main_pretrain.py --config-path config --config-name pretrain/pretrain.yaml
   ```

**General knowledge transfer** - The knowledge transfer experiments can be run using the following command:
   ```python
    python main_transfer.py --config-path config --config-name transfer/transfer.yaml
   ```

**Multi-teacher knowledge transfer** - The multi-teacher knowledge transfer experiments can be run using the following command:
   ```python
    python main_multi_teacher.py --config-path config --config-name multi_teacher/multi_teacher.yaml
   ```

## Citation
If you find this work useful, please consider citing the following paper:
```
@misc{roth2024fantasticgainsthemexistence,
      title={Fantastic Gains and Where to Find Them: On the Existence and Prospect of General Knowledge Transfer between Any Pretrained Model}, 
      author={Karsten Roth and Lukas Thede and Almut Sophia Koepke and Oriol Vinyals and Olivier Hénaff and Zeynep Akata},
      year={2024},
      eprint={2310.17653},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.17653}, 
}
```


