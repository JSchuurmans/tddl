# Logging experiments

## 4 Feb
Training ResNet-18 baseline

Train ResNet-18 like FMix paper with batchsize 128 for 200 epochs with Adam an a learning rate of 0.1, that is multiplied with 0.1 at epoch 100 and 150.
```
python src/tddl/f_mnist.py main --config-path configs/tud/f_mnist/train_fmix.yml
```

