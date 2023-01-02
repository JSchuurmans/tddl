# TDDL
Tensor Decomposition for Deep Learning


## Install Guide

### Install pytorch 
```bash
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install tddl 
```
pip install -e ."[dev]"
```


## Getting started

    from tddl.factorizations import factorize_network

    factorize_network(      
        model,                  # Changes your pytorch model inplace.
        layers,                 # Modifies only layers (or layer type) you specify,
        factorization='tucker', # into specific factorization,
        rank=0.5,               # with a given (fractional) rank.
        decompose_weights=True, # Decompose the weights of the model.
    )

To know how the layers are numbered, we provide the utility function `number_layers`.

    from tddl.factorizations import number_layers

    number_layers(model)


## Hyperparameter Tuning with RayTune
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html


`python -m train.py --config-path configs/tune.yml`

## Docker

In the commands below:

- \`pwd\` adds the path to the current directory. Make sure you are in the tddl folder where this repo is in. This adds the config files to the docker container.
- `/source/to/data:/destination/to/data`: mounts the data directory to a destination inside the docker container. Modify this to your data path. E.g. we use `/bigdata:/bigdata` and have in the config files the data and log directories somewhere in `/bigdata/`, data:`/bigdata/cifar10/` logs: `/bigdata/cifar10/logs`.

### Build yourself
```
docker build . -t tddl
```

```
docker run -it -v `pwd`:`pwd` -v /source/to/data:/destination/to/data  --gpus all tddl
```

### Pull from DockerHub
```
docker pull jtsch/tddl:latest
```

```
docker run -it -v `pwd`:`pwd` -v /source/to/data:/destination/to/data  --gpus all jtsch/tddl:latest
```

# Reproduce 

The results are produced with Python 3.8.10, GCC 9.4.0, Ubuntu 20.04, CUDA 11.1, cuDNN 8.1.1.
For specific package versions see the `requirements.txt`. Note that the `requirements.txt` is only used for documenting the versions and not the installation. Check `Install Guide` for instructions for installation.

## ICLR 2023

Links:
- [Latex](https://www.overleaf.com/project/6321df4c381ffd48c08a027f)
- [PDF](papers/iclr_2023/Schuurmans et al (2023) How informative is the approximation error.pdf)
- [OpenReview](https://openreview.net/forum?id=sKHqgFOaFXI&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions))

### Train baseline model

#### Pretrain ResNet-18 on CIFAR-10
```bash
python train.py main --config-path papers/iclr_2023/configs/tud/cifar10/train_rn18.yml
```

#### Pretrain GaripovNet on CIFAR-10
```bash
python train.py main --config-path papers/iclr_2023/configs/tud/garipov/cifar10/train_garipov.yml
```

#### Pretrain GaripovNet on F-MNIST
```bash
python train.py main --config-path papers/iclr_2023/configs/tud/garipov/fmnist/train_garipov.yml
```

### Factorize and Fine-tune
Make sure the path to pretrained model is provided in config files. 

#### Factorize with CP and Tucker and Fine-tune ResNet-18 on CIFAR-10
```bash
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 1 25 5 75 9; 
do for FACT in cp tucker;
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path papers/icrl_2023/configs/rn18/cifar10/decompose/dec-$FACT-r0.5-$LAYER.yml --data-workers=8 --rank=0.$RANK; 
done;
done;
done;
done
```

#### Factorize with TT and Fine-tune ResNet-18 on CIFAR-10 
```bash
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 1 25 5 75 9; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path papers/icrl_2023/configs/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4; 
done;
done;
done
```


#### Factorize with CP and Tucker and Fine-tune GaripovNet on CIFAR-10 
```bash
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do for FACT in cp tucker;    
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path papers/icrl_2023/configs/garipov/fmnist/decompose/dec-cp-r0.5-$LAYER.yml --rank=0.$RANK --factorization=$FACT; 
done;
done;
done;
done
```

#### Factorize with TT and Fine-tune Garipov on CIFAR-10
```bash
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do echo "{$i}-{$LAYER}-{$RANK}" && python train.py main --config-path papers/icrl_2023/configs/garipov/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4;
done;
done;
done
```

#### Factorize with TT and Fine-tune GaripovNet on F-MNIST

```bash
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do echo "{$i}-{$LAYER}-{$RANK}" && python train.py main --config-path papers/icrl_2023/configs/garipov/fmnist/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4;
done;
done;
done
```


### Calculate Error on features

#### ResNet-18 on CIFAR-10 for training dataset
```bash
python src/tddl/features/extract.py main /bigdata/cifar10/logs/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 8
```

#### GaripovNet on CIFAR-10 for training dataset
```bash
python src/tddl/features/extract.py main /bigdata/cifar10/logs/garipov/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 8
```

#### GaripovNet on F-MNIST for training dataset
```bash
python src/tddl/features/extract.py main /bigdata/f_mnist/logs/garipov/decomposed --dataset fmnist --split train --aggregate --skip-existing --data-workers 8
```


### Post process

#### process_factorized_networks

```bash
python src/tddl/post_processing/factorized_model.py --logdir /bigdata/cifar10/logs/decomposed --baseline-path /bigdata/cifar10/logs/baselines/1646668631/rn18_18_dNone_128_adam_l0.001_g0.1_w0.0_sTrue
```

```bash
python src/tddl/post_processing/factorized_model.py --logdir /bigdata/cifar10/logs/garipov/decomposed/ --baseline_path /bigdata/cifar10/logs/garipov/baselines/1647358615/gar_18_dNone_128_sgd_l0.1_g0.1_w0.0_sTrue
```

```bash
python src/tddl/post_processing/factorized_model.py --logdir /bigdata/f_mnist/logs/garipov/decomposed/ --baseline_path /bigdata/f_mnist/logs/garipov/baselines/1647955843/gar_18_dNone_128_sgd_l0.1_g0.1_w0.0_sTrue
```


#### Create Tables

```bash
python src/tddl/post_processing/create_tables.py \
--logdir /bigdata/cifar10/logs/decomposed \
--output papers/iclr_2023/tables/rn18/cifar10/ \
--tt-conversion papers/iclr_2023/configs/rn18/rn18_tt_actual_rank_to_tl_ranks.json \
--model rn18 \
--dataset c10
```

```bash
python src/tddl/post_processing/create_tables.py \
--logdir /bigdata/cifar10/logs/garipov/decomposed \
--output papers/iclr_2023/tables/gar/cifar10/ \
--tt-conversion papers/iclr_2023/configs/garipov/gar_tt_actual_rank_to_tl_ranks.json \
--model gar \
--dataset c10
```

```bash
python src/tddl/post_processing/create_tables.py \
--logdir /bigdata/f_mnist/logs/garipov/decomposed \
--output papers/iclr_2023/tables/gar/f_mnist/ \
--tt-conversion papers/iclr_2023/configs/garipov/gar_tt_actual_rank_to_tl_ranks.json \
--model gar \
--dataset fm
```

### Create plots
Run notebook:
`papers/iclr_2023/notebooks/results/kendalls_tau.ipynb`


## [WIP] Double Binary Search (DBS)

Link to latex: https://www.overleaf.com/project/6397024bb070ec521aadb28d

Use the configs in: `papers/dbs/configs`

## [WIP] Decompose vs Factorize (DVSF)

Link to latex: https://www.overleaf.com/project/6179366dd37ad23166523d27

Use the configs in: `papers/dvsf/configs`

To factorize a model (in this case layer 15) and train it run: 
```bash
python train.py main --config-path papers/dvsf/configs/rn18/cifar10/factorize/fac-tucker-r0.5-15.yml
```

To train a baseline model:
```bash
python train.py main --config-path papers/dvsf/configs/rn18/cifar10/train_baseline.yml
```

To decompose (in this case layer 15 of) the baseline model modify the path to the baseline model and run:
```bash
python train.py main --config-path papers/dvsf/configs/rn18/cifar10/decompose/dec-tucker-r0.5-15.yml
```
