from time import time
import copy
from pathlib import Path
from typing import List

import typer
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from tddl.models.vgg import ModifiedVGG16Model
from tddl.trainer import Trainer
import tensorly as tl
import tltorch
from torchsummary import summary

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from tddl.models.cnn import Net, TdNet

from tddl.utils.prime_factors import get_prime_factors

app = typer.Typer()

transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

dataset = datasets.MNIST('/bigdata/mnist', train=True, download=True, transform=transform)
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, (50000, 10000), generator=torch.Generator().manual_seed(42))
test_dataset = datasets.MNIST('/bigdata/mnist', train=False, transform=transform)



@app.command()
def train(
    batch: int = 256,
    epochs: int = 20,
    logdir="/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/mnist",
    lr: float = 0.001,
    gamma: float = 0.9,
    conv1_out: int = 32,
    conv2_out: int = 64,
):

    t = round(time())
    MODEL_NAME = f"cnn-{conv1_out}-{conv2_out}_bn_{batch_size}_adam_l{lr}_g{gamma}"
    logdir = Path(logdir).joinpath(MODEL_NAME,str(t))
    save = {
        "save_every_epoch": 10,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": "cnn"
    }

    # TODO add data augmentation
    # Cutout/random crop (DeVries, 2017)
    # Mixup (Zhang, ICLR 2018)
    # salt and pepper
    # rotation
    # translation
    # streching
    # shearing
    # lens distortions
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch)
    # test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))
    model = Net(conv1_out=conv1_out, conv2_out=conv2_out).cuda()
    # TODO why not Adam ?
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    trainer = Trainer(train_loader, valid_loader, model, optimizer, writer, scheduler=scheduler, save=save)
    trainer.train(epochs=epochs)

    writer.close()


@app.command()
def decompose(
    pretrained: str = "/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/mnist/cnn-32-32_bn_256_adam_l0.01_g0.9/1628155584/cnn_best",
    layer_nrs: List[int] = 0,
    factorization: str = 'tucker',
    decompose_weights: bool = True,
    td_init: float = 0,
    rank: float = 0.5,
    epochs: int = 10,
    lr: float = 1e-2,
    logdir: str = "/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/mnist",
    freeze_parameters: bool = False,
    batch: int = 256,
    gamma: float = 0.9,
):

    model = torch.load(pretrained)
    fact_model = copy.deepcopy(model)
    
    # which parameters to train
    # if freeze_parameters:
    #     for param in fact_model.parameters():
    #         param.requires_grad = False

    if decompose_weights:
        td_init = False

    # layer_nrs = [2,6] if layer_nr == 0 else [layer_nr]

    for i, (name, module) in enumerate(model.named_modules()):
        if i in layer_nrs:
            if type(module) == torch.nn.modules.conv.Conv2d:
                fact_layer = tltorch.FactorizedConv.from_conv(
                    module, 
                    rank=rank, 
                    decompose_weights=decompose_weights, 
                    factorization=factorization
                )
            elif type(module) == torch.nn.modules.linear.Linear:
                fact_layer = tltorch.FactorizedLinear.from_linear(
                    module, 
                    in_tensorized_features=get_prime_factors(module.in_features), 
                    out_tensorized_features=get_prime_factors(module.out_features), 
                    rank=rank,
                    factorization=factorization,
                )
            if td_init:
                fact_layer.weight.normal_(0, td_init)
            fact_model._modules[name] = fact_layer
    print(fact_model)

    MODEL_NAME = f"td-{layer_nr}-{factorization}-{rank}-d{str(decompose_weights)}-i{td_init}_bn_{batch}_adam_l{lr}_g{gamma}"
    t = round(time())
    logdir = Path(logdir).joinpath(MODEL_NAME,str(t))
    save = {
        "save_every_epoch": 1,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": f"fact_model"
    }
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, fact_model.parameters()),
        lr=lr
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma) if gamma else None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch)

    trainer = Trainer(train_loader, valid_loader, model, optimizer, writer, scheduler=scheduler, save=save)
    trainer.train(epochs=epochs)
    
    train_acc = trainer.test(loader="train")
    writer.add_scalar("Accuracy/before_finetuning/train", train_acc)
    valid_acc = trainer.test()
    writer.add_scalar("Accuracy/before_finetuning/valid", valid_acc)

    trainer.train(epochs=epochs)

    writer.close()


@app.command()
def low_rank(
    layer_nrs: List[int],
    factorization: str = 'tucker',
    td_init: float = 0.02,
    rank: float = 0.5,
    epochs: int = 20,
    lr: float = 1e-2,
    logdir: str = "/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/mnist",
    batch: int = 256,
    gamma: float = 0.9,
    conv1_out: int = 32,
    conv2_out: int = 32,
    fc1_out: int = 128
):
    model = TdNet(
        conv1_out=conv1_out, conv2_out=conv2_out, fc1_out=fc1_out, 
        layer_nrs=layer_nrs, rank=rank, factorization=factorization, td_init=td_init,
    ).cuda()

    print(model)

    MODEL_NAME = f"lr-{conv1_out}-{conv2_out}-{layer_nrs}-{factorization}-{rank}-i{td_init}_bn_{batch}_adam_l{lr}_g{gamma}"
    t = round(time())
    logdir = Path(logdir).joinpath(MODEL_NAME,str(t))
    save = {
        "save_every_epoch": 1,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": f"fact_model"
    }
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma) if gamma else None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch)

    trainer = Trainer(train_loader, valid_loader, model, optimizer, writer, scheduler=scheduler, save=save)
    trainer.train(epochs=epochs)

    writer.close()


if __name__ == "__main__":
    app()
