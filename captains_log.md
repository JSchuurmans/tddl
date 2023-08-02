# Logging experiments

## 4 Feb
Training ResNet-18 baseline

Train ResNet-18 like FMix paper with batchsize 128 for 200 epochs with Adam an a learning rate of 0.1, that is multiplied with 0.1 at epoch 100 and 150:
```
python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/train_fmix.yml
```

Train ResNet-18 like FMix paper with batchsize 128 for 300 epochs with SGD and momentum of 1e-4 an a learning rate of 0.1, that is multiplied with 0.1 at epoch 100 and 225:
```
python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/train_re.yml
```

Train ResNet-18 like FMix paper with batchsize 128 for 300 epochs with SGD and momentum of 1e-4 an a learning rate of 0.1, that is multiplied with 0.1 at epoch 100 and 225:
```
python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/train_re.yml --weight-decay 1.0e-4
```

## 7 Feb

Use for loop in bash to run multiple config files:
```
for LAYER in 15 19 28 38 41 44 60 63; 
do for FACT in cp tucker; 
do python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/decompose/dec-$FACT-r0.5-$LAYER.yml;
done;
done
```

## 14 Feb

Use for loop in bash to run multiple config files and do additional loop for error bars:
```
for i in {1..5};
do for LAYER in 15 19 38 41 44 60 63; 
do for FACT in cp tucker; 
do echo "{$i}-{$LAYER}-{$FACT}" && python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/decompose/sgd/dec-$FACT-r0.5-$LAYER.yml;
done;
done;
done
```

```
for i in {1..5};
do for RANK in 1 25 50 75 90; 
do for FACT in cp tucker; 
do echo "{$i}-{$RANK}-{$FACT}" && python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/decompose/sgd/dec-$FACT-r0.5-28.yml --rank=0.$RANK;
done;
done;
done
```

## 16 Feb
Run with Adam
```
for i in {1..5};
do for LAYER in 15 19 38 41 44 60 63; 
do for FACT in cp tucker; 
do echo "{$i}-{$LAYER}-{$FACT}" && python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/decompose/adam/dec-$FACT-r0.5-$LAYER.yml;
done;
done;
done
```


```
for i in {1..5};
do for RANK in 1 25 50 75 90; 
do for FACT in cp tucker; 
do echo "{$i}-{$RANK}-{$FACT}" && python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/decompose/adam/dec-$FACT-r0.5-28.yml --rank=0.$RANK;
done;
done;
done
```

## 21 Feb
Fix decomposition, vary training seed

```
for i in {1..5};
do for LAYER in 15 19 38 41 44 60 63; 
do echo "{$i}-{$LAYER}" && python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/fixed_decomp/adam/dec-cp-r0.5-$LAYER.yml --logdir="/bigdata/f_mnist/logs/fixed_decomp";
done;
done
```


```
for i in {1..5};
do for RANK in 1 25 5 75 9; 
do echo "{$i}-{$RANK}" && python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/fixed_decomp/adam/dec-cp-r0.$RANK-28.yml --logdir="/bigdata/f_mnist/logs/fixed_decomp";
done;
done
```

## 1 Mar

python train.py main --config-path configs/tud/cifar10/train_rn18.yml 

python train.py main --config-path configs/tud/cifar10/train_rn18.yml --optimizer=adam

python train.py main --config-path configs/tud/cifar10/train_rn18.yml --lr=0.01

python train.py main --config-path configs/tud/cifar10/train_rn18.yml --lr=0.01 --optimizer=adam

## 4 Mar

python train.py main --config-path configs/tud/cifar10/train_rn18.yml --lr=1

python train.py main --config-path configs/tud/cifar10/train_rn18.yml --lr=0.001 --optimizer=adam


### TODO

no weight decay for sgd

python train.py main --config-path configs/tud/cifar10/train_rn18.yml 

no milestones

try batch size

more epochs

1646401923/rn18_18_dNone_128_adam_l0.001_g0.1_sTrue/runs

has no regularization in the form of weight_decay
is/should be 0 during fine-tuning

## 7 Mar

1646668631/rn18_18_dNone_128_adam_l0.001_g0.1_w0.0_sTrue/runs

```
for i in {1..5};
do for LAYER in 15 19 38 41 44 60 63; 
do for FACT in cp tucker; 
do echo "{$i}-{$LAYER}-{$FACT}" && python train.py main --config-path configs/tud/cifar10/decompose/dec-$FACT-r0.5-$LAYER.yml;
done;
done;
done
```

```
for i in {1..5};
do for RANK in 1 25 50 75 90; 
do for FACT in cp tucker; 
do echo "{$i}-{$RANK}-{$FACT}" && python train.py main --config-path configs/tud/cifar10/decompose/dec-$FACT-r0.5-28.yml --rank=0.$RANK;
done;
done;
done
```

## 12 Mar

```
for i in {1..5};
do for LAYER in 15 19 38 41 44 60 63;
do for RANK in 1 25 
do for FACT in cp tucker; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/cifar10/decompose/dec-$FACT-r0.5-$LAYER.yml --rank=0.$RANK;
done;
done;
done;
done
```

```
for i in {1..5};
do for LAYER in 15 19 38 41 44 60 63;
do for RANK in 1 25 
do for FACT in cp tucker; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/f_mnist/decompose/adam/dec-$FACT-r0.5-$LAYER.yml --rank=0.$RANK;
done;
done;
done;
done
```

## 14 Mar

`python src/tddl/features/extract.py main /bigdata/cifar10/logs/decomposed --dataset cifar10 --split train --aggregate --skip-existing`


same for f_mnist

## 15 Mar

`python train.py main --config-path configs/tud/garipov/cifar10/train_garipov.yml`

`python train.py main --config-path configs/tud/garipov/cifar10/train_garipov.yml --batch=256`

## 20 Mar

```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do for FACT in cp tucker; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/decompose/dec-$FACT-r0.5-$LAYER.yml --rank=0.$RANK;
done;
done;
done;
done
```
**Tucker FAILED: no config**
```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do for FACT in tucker;    
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/decompose/dec-cp-r0.5-$LAYER.yml --rank=0.$RANK --factorization=$FACT; 
done;
done;
done;
done
```

## 21 Mar
python src/tddl/features/extract.py main /bigdata/cifar10/logs/garipov/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 8

python src/tddl/features/extract.py main /bigdata/cifar10/logs/garipov/decomposed --dataset cifar10 --split valid --aggregate --skip-existing --data-workers 8


## 22 Mar

`python train.py main --config-path configs/tud/garipov/fmnist/train_garipov.yml`

```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do for FACT in cp tucker;    
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/garipov/fmnist/decompose/dec-cp-r0.5-$LAYER.yml --rank=0.$RANK --factorization=$FACT; 
done;
done;
done;
done
```

## 23 Mar
`python src/tddl/features/extract.py main /bigdata/f_mnist/logs/garipov/decomposed --dataset fmnist --split train --aggregate --skip-existing --data-workers 8`

`python src/tddl/features/extract.py main /bigdata/f_mnist/logs/garipov/decomposed --dataset fmnist --split valid --aggregate --skip-existing --data-workers 4`







## 20 Jun

### Run RN18 on CIFAR10 with TT

```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 1 25 5; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/decompose/dec-tt-r0.5-$LAYER.yml --rank=0.$RANK --data-workers=4; 
done;
done;
done
```

### Run GaripovNet on CIFAR10 with TT

```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do for FACT in tt;
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/decompose/dec-cp-r0.5-$LAYER.yml --rank=0.$RANK --factorization=$FACT --data-workers=4; 
done;
done;
done;
done
```

### Run GaripovNet on F-MNIST with TT

```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do for FACT in tt;
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/garipov/fmnist/decompose/dec-cp-r0.5-$LAYER.yml --rank=0.$RANK --factorization=$FACT --data-workers=4; 
done;
done;
done;
done
```

## 21 Juni

`python src/tddl/features/extract.py main /bigdata/f_mnist/logs/garipov/decomposed --dataset fmnist --split train --aggregate --skip-existing --data-workers 4`

`python src/tddl/features/extract.py main /bigdata/cifar10/logs/garipov/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 4`

Move from `/bigdata/cifar10/logs/decomposed/tt/` to `/bigdata/cifar10/logs/decomposed/`

`python src/tddl/features/extract.py main /bigdata/cifar10/logs/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 4`


## 15 Aug

### Move old TT - 
#### ResNet CIFAR-10
`cd /bigdata/cifar10/logs/decomposed`

`rm -rf 1655* `

#### CIFAR-10 GaripovNet
`cd /bigdata/cifar10/logs/garipov/decomposed`

`rm -rf 1655* `

#### FMNIST GaripovNet
`cd /bigdata/f_mnist/logs/garipov/decomposed`

`rm -rf 1655* `

### TT ResNet CIFAR-10

```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 1; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4; 
done;
done;
done
```

```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 25; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4; 
done;
done;
done
```

```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 5; 
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4; 
done;
done;
done
```



### Garipov CIFAR-10 with TT
```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do echo "{$i}-{$LAYER}-{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4;
done;
done;
done
```

### Run GaripovNet on F-MNIST with TT

```
for i in {1..5};
do for LAYER in 2 4 6 8 10;
do for RANK in 1 25 5 75 9;
do echo "{$i}-{$LAYER}-{$RANK}" && python train.py main --config-path configs/tud/garipov/fmnist/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=4;
done;
done;
done
```

## 16 Aug
`python src/tddl/features/extract.py main /bigdata/f_mnist/logs/garipov/decomposed --dataset fmnist --split train --aggregate --skip-existing --data-workers 4`

`python src/tddl/features/extract.py main /bigdata/cifar10/logs/garipov/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 4`

`python src/tddl/features/extract.py main /bigdata/cifar10/logs/decomposed --dataset cifar10 --split train --aggregate --skip-existing --data-workers 4`

# Train factorized model from random init

`python train.py main --config-path configs/tud/rn18/cifar10/factorized/fac-tucker-r0.5-15.yml`


```
for i in {1..5};
do for LAYER in 4;
do for RANK in 5;
do echo "{$i}-{$LAYER}-{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=6;
done;
done;
done
```

```
for i in {1..5};
do for LAYER in 4;
do for RANK in 25;
do echo "{$i}-{$LAYER}-{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=6;
done;
done;
done
```

## 21 Sept
```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 75 9; 
do for FACT in cp;
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/decompose/dec-tucker-r0.5-$LAYER.yml --data-workers=8 --rank=0.$RANK; 
done;
done;
done;
done
```

```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 75 9;
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/decompose/dec-cp-r0.5-$LAYER.yml --data-workers=8 --rank=0.$RANK; 
done;
done;
done
```

```
for i in {1..5};
do for LAYER in 15 19 28 38 41 44 60 63;
do for RANK in 75 9;
do echo "{$i}-{$LAYER}-{$FACT}-{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/decompose/dec-tt-r0.$RANK-$LAYER.yml --data-workers=8; 
done;
done;
done
```

# Fine-tune all layers with different compression levels

## 15 Dec

`python train.py main --config-path configs/tud/garipov/cifar10/dbs/dbs-tucker-r0.5.yml`

```bash
for RANK in 1 75 9;
do echo "{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/dbs/dbs-tucker-r0.$RANK.yml; 
done
```

```bash
for RANK in 1 75 9;
do echo "{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/dbs/dbs-tucker-r0.$RANK.yml-constant; 
done
```

```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/dbs/dbs-tucker-r0.1.yml --rank=0.$RANK; 
done
```

```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/dbs/dbs-tucker-r0.1-constant.yml --rank=0.$RANK; 
done
```

## 16 Dec

```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/dbs/dbs-cp-r0.5.yml --rank=0.$RANK; 
done
```

```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path configs/tud/garipov/cifar10/dbs/dbs-cp-r0.5-constant.yml --rank=0.$RANK; 
done
```


```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/dbs/dbs-cp-r0.1.yml --rank=0.$RANK; 
done
```

```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path configs/tud/rn18/cifar10/dbs/dbs-cp-r0.1-constant.yml --rank=0.$RANK; 
done
```

# 2023

## 20 Jan

### Pretrain ResNet-18 on CIFAR-10
```bash
python train.py main --config-path papers/dbs/configs/rn18/cifar10/train_baseline.yml --data-workers=8
```

### Test tucker with logging time
python train.py main --config-path papers/dbs/configs/rn18/cifar10/dbs-tucker-r0.1.yml --rank=0.5 --logdir="/bigdata/cifar10/logs/rn18/dbs/tmp/tucker"

### Test cp with logging time
python train.py main --config-path papers/dbs/configs/rn18/cifar10/dbs-cp-r0.1.yml --rank=0.5 --logdir="/bigdata/cifar10/logs/rn18/dbs/tmp/cp"

### Run DBS Tucker with new baseline
```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path papers/dbs/configs/rn18/cifar10/dbs-tucker-r0.1.yml --rank=0.$RANK --logdir="/bigdata/cifar10/logs/rn18/dbs/new_baseline/tucker"; 
done
```

### Run constant compression with Tucker and new baseline
```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path papers/dbs/configs/rn18/cifar10/dbs-tucker-r0.1-constant.yml --rank=0.$RANK --logdir="/bigdata/cifar10/logs/rn18/constant/new_baseline/tucker"; 
done
```

### Run DBS CP with new baseline
```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path papers/dbs/configs/rn18/cifar10/dbs-cp-r0.1.yml --rank=0.$RANK --logdir="/bigdata/cifar10/logs/rn18/dbs/new_baseline/cp"; 
done
```

### Run constant compression with CP and new baseline
```bash
for RANK in 1 25 5 75 9;
do echo "{$RANK}" && python train.py main --config-path papers/dbs/configs/rn18/cifar10/dbs-cp-r0.1-constant.yml --rank=0.$RANK --logdir="/bigdata/cifar10/logs/rn18/constant/new_baseline/cp"; 
done
```
