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
