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

