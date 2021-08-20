from time import time
import copy
from pathlib import Path
from typing import List

import typer
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tddl.trainer import Trainer
import tensorly as tl
import tltorch
from torchsummary import summary

from torchvision import datasets, transforms
from torch.optim import lr_scheduler

from tddl.models.wrn import WideResNet
from tddl.utils.prime_factors import get_prime_factors
from tddl.data.sets import DatasetFromSubset

app = typer.Typer()

transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.Normalize((0.1307,), (0.3081,)),
])

dataset = datasets.FashionMNIST('/bigdata/f_mnist', train=True, download=True)
train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset,
    (50000, 10000),
    generator=torch.Generator().manual_seed(42)
)
train_dataset = DatasetFromSubset(
    train_dataset, transform=transform_train,
)
valid_dataset = DatasetFromSubset(
    train_dataset, transform=transform_test,
)

# test_dataset = datasets.FashionMNIST('/bigdata/f_mnist', train=False, transform=transform_test)



@app.command()
def train(
    batch: int = 256,
    epochs: int = 300,
    logdir: str ="/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/f_mnist",
    lr: float = 0.1,
    gamma: float = 0.1,
    dropout: float = 0.5,
    model_name: str = "wrn",
    depth: int = 28,
    width: int = 10,
):

    logdir = Path(logdir)
    if not logdir.is_dir():
        raise FileNotFoundError("{0} folder does not exist!".format(l))
    t = round(time())
    MODEL_NAME = f"{model_name}_{depth}_d{dropout}_{batch}_sgd_l{lr}_g{gamma}"
    logdir = logdir.joinpath(MODEL_NAME,str(t))
    save = {
        "save_every_epoch": 10,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": "cnn"
    }

    # TODO add data augmentation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch)
    # test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))

    model = WideResNet(
        depth=depth,
        num_classes=10,
        widen_factor=10,
        dropRate=dropout,
    ).cuda()

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 225], gamma=gamma)
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

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
