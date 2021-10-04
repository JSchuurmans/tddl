import json
import os
from time import time
from pathlib import Path
from typing import List

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tddl.data.loaders import get_f_mnist_loader
from tddl.trainer import Trainer
from tddl.models.wrn import WideResNet
from tddl.models.resnet import PA_ResNet18
from tddl.models.resnet_lr import low_rank_resnet18
from tddl.utils.random import set_seed
from tddl.models.utils import count_parameters

import typer

app = typer.Typer()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


@app.command()
def train(
    batch: int = 256,
    epochs: int = 200,
    logdir: Path = Path("/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/f_mnist"),
    lr: float = 0.1,
    gamma: float = 0.1,
    dropout: float = 0.5,
    model_name: str = "parn",
    depth: int = 18,
    width: int = 10,
    data_workers: int = 1,
    seed: int = None,
    data: Path = Path("/bigdata/f_mnist"),
):

    # logdir = Path(logdir)
    if not logdir.is_dir():
        raise FileNotFoundError("{0} folder does not exist!".format(logdir))
    t = round(time())
    if seed is None:
        seed = t
    set_seed(seed)
    MODEL_NAME = f"{model_name}_{depth}_d{dropout}_{batch}_sgd_l{lr}_g{gamma}_s{seed == t}"
    logdir = logdir.joinpath(MODEL_NAME,str(t))
    save = {
        "save_every_epoch": None,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": "cnn"
    }

    train_dataset, valid_dataset, test_dataset = get_f_mnist_loader(data)

    # TODO add data augmentation
    train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=data_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch, num_workers=data_workers)
    # test_loader = DataLoader(train_dataset, batch_size=batch_size)
    
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))

    num_classes = 10
    if model_name == 'wrn':
        model = WideResNet(
            depth=depth,
            num_classes=num_classes,
            widen_factor=width,
            dropRate=dropout,
        ).cuda()
        milestones = [100, 150, 225]
    elif model_name == "parn":
        model = PA_ResNet18(
            num_classes=num_classes, 
            nc=1,
        ).cuda()
        milestones = [100, 150]

    n_param = count_parameters(model)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    trainer = Trainer(train_loader, valid_loader, model, optimizer, writer, scheduler=scheduler, save=save)
    results = trainer.train(epochs=epochs)

    results['n_param'] = n_param
    results['model_name'] = MODEL_NAME
    with open(logdir.joinpath('results.json'), 'w') as f:
        json.dump(results, f)

    writer.close()


@app.command()
def decompose(
    layers: List[int],
    pretrained: str = "/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/f_mnist/parn_18_d0.5_256_sgd_l0.1_g0.1/1629473591/cnn_best",
    factorization: str = 'tucker',
    # decompose_weights: bool = True,
    td_init: float = 0.02,
    rank: float = 0.5,
    epochs: int = 200,
    lr: float = 0.1,
    logdir: Path = Path("/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/f_mnist"),
    # freeze_parameters: bool = False,
    batch: int = 256,
    gamma: float = 0,
    model_name: str = "parn",
    seed: int = None,
    data_workers: int = 1,
    data: Path = Path("/bigdata/f_mnist")
):

    if pretrained == "":
        model = None
        decompose_weights = False
    else:
        model = torch.load(pretrained)
        decompose_weights = True

    if decompose_weights:
        td_init = False

    fact_model = low_rank_resnet18(
        layers=layers,
        rank=rank,
        decompose_weights=decompose_weights,
        factorization=factorization,
        init=td_init,
        pretrained_model=model,
    ).cuda()

    n_param = count_parameters(fact_model)
    
    if not logdir.is_dir():
        raise FileNotFoundError("{0} folder does not exist!".format(logdir))
    td = "td" if pretrained != "" else "lr"
    t = round(time())
    if seed is None:
        seed = t
    set_seed(seed)
    MODEL_NAME = f"{model_name}-{td}-{layers}-{factorization}-{rank}-d{str(decompose_weights)}-i{td_init}_bn_{batch}_sgd_l{lr}_g{gamma}_s{seed == t}"
    logdir = logdir.joinpath(MODEL_NAME, str(t))
    save = {
        "save_every_epoch": None,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": f"fact_model"
    }
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))

    # TODO change back to Adam
    # optimizer = optim.Adam(
    #     filter(lambda p: p.requires_grad, fact_model.parameters()),
    #     lr=lr
    # )
    optimizer = optim.SGD(
        fact_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) if gamma else None
    if pretrained == "":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=gamma)

    train_dataset, valid_dataset, test_dataset = get_f_mnist_loader(data)

    train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=data_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch, num_workers=data_workers)

    trainer = Trainer(train_loader, valid_loader, fact_model, optimizer, writer, scheduler=scheduler, save=save)
    
    if pretrained != "":
        train_acc = trainer.test(loader="train")
        writer.add_scalar("Accuracy/before_finetuning/train", train_acc)
        valid_acc = trainer.test()
        writer.add_scalar("Accuracy/before_finetuning/valid", valid_acc)

    results = trainer.train(epochs=epochs)
    
    results['model_name'] = MODEL_NAME
    results['n_param_fact'] = n_param
    if pretrained != "":
        results['approx_error'] = None # TODO
        results['train_acc_before_ft'] = train_acc
        results['valid_acc_before_ft'] = valid_acc
    with open(logdir.joinpath('results.json'), 'w') as f:
        json.dump(results, f)

    writer.close()


if __name__ == "__main__":
    app()
