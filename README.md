# TDDL
Tensor Decomposition for Deep Learning


## Install Guide

Install pytorch `pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

Install tddl `pip install -e ."[dev]"`


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




## Reproduce


### Train baseline model

#### Pretrain ResNet-18 on CIFAR-10
`python train.py main --config-path configs/cifar10/train_rn18.yml`

#### Pretrain GaripovNet on CIFAR-10
`python train.py main --config-path configs/garipov/cifar10/train_garipov.yml`

#### Pretrain GaripovNet on F-MNIST
`python train.py main --config-path configs/garipov/fmnist/train_garipov.yml`



### Factorize and Fine-tune
Make sure the path to pretrained model is provided in config files. 

#### Factorize with CP and Tucker and Fine-tune ResNet-18 on CIFAR-10
```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 1 25 5 75 9; 
do for FACT in cp tucker;
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/rn18/cifar10/decompose/dec-$FACT-r0.5-$LAYER.yml --data-workers=8 --rank=0.$RANK; 
done;
done;
done;
done
```

#### Factorize with TT and Fine-tune ResNet-18 on CIFAR-10 
```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 1 25 5 75 9; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4; 
done;
done;
done
```


#### Factorize with CP and Tucker and Fine-tune GaripovNet on CIFAR-10 
```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do for FACT in cp tucker;    
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/garipov/fmnist/decompose/dec-cp-r0.5-$LAYER.yml --rank=0.$RANK --factorization=$FACT; 
done;
done;
done;
done
```

#### Factorize with TT and Fine-tune Garipov on CIFAR-10
```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do echo "{$i}-{$LAYER}-{$RANK}" && python train.py main --config-path configs/garipov/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4;
done;
done;
done
```

#### Factorize with TT and Fine-tune GaripovNet on F-MNIST

```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do echo "{$i}-{$LAYER}-{$RANK}" && python train.py main --config-path configs/garipov/fmnist/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4;
done;
done;
done
```


### Calculate Error on features

#### ResNet-18 on CIFAR-10 for training dataset
`python src/tddl/features/extract.py main /bigdata/cifar10/logs/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 4`

#### GaripovNet on CIFAR-10 for training dataset
`python src/tddl/features/extract.py main /bigdata/cifar10/logs/garipov/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 8`

#### GaripovNet on F-MNIST for training dataset
`python src/tddl/features/extract.py main /bigdata/f_mnist/logs/garipov/decomposed --dataset fmnist --split train --aggregate --skip-existing --data-workers 8`


### Post process
Run notebooks:
- `notebooks/results/rn18_c10_approx_vs_perform_ci.ipynb`
- `notebooks/results/gar_c10_appro_vs_perform_ci.ipynb`
- `notebooks/results/gar_fm_appro_vs_perform_ci.ipynb`


### Create plots
Run notebook:
`notebooks/results/kendalls_tau.ipynb`
