import json
import os
from time import time
from pathlib import Path
from typing import List
from functools import partial

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from tddl.data.loaders import get_f_mnist_loader
from tddl.trainer import Trainer
from tddl.models.wrn import WideResNet
from tddl.models.resnet import PA_ResNet18
from tddl.models.resnet_lr import low_rank_resnet18
from tddl.utils.random import set_seed
from tddl.models.utils import count_parameters

import tensorly as tl

import typer

app = typer.Typer()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

tl.set_backend('pytorch')

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
    cpu: int = None,
    data_workers: int = 1,
    seed: int = None,
    data_dir: Path = Path("/bigdata/f_mnist"),
    cuda: str = None,
) -> None:

    if cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    
    if cpu is not None:
        os.environ["MKL_NUM_THREADS"] = cpu
        os.environ["NUMEXPR_NUM_THREADS"] = cpu
        os.environ["OMP_NUM_THREADS"] = cpu
        data_workers = cpu # max(cpu-1, 1)

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

    train_dataset, valid_dataset, test_dataset = get_f_mnist_loader(data_dir)

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
    data_dir: Path = Path("/bigdata/f_mnist"),
    cuda: str = None,
    cpu: str = None,
    checkpoint_dir: str = None,
    config: str = None,
) -> None:

    if cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    if cpu is not None:
        os.environ["MKL_NUM_THREADS"] = cpu
        os.environ["NUMEXPR_NUM_THREADS"] = cpu
        os.environ["OMP_NUM_THREADS"] = cpu
        data_workers = int(cpu) # max(int(cpu)-1, 1)

    if pretrained == "":
        model = None
        decompose_weights = False
    else:
        model = torch.load(pretrained)
        decompose_weights = True

    if decompose_weights:
        td_init = False

    if config is not None:
        lr = config['lr']

    model = low_rank_resnet18(
        layers=layers,
        rank=rank,
        decompose_weights=decompose_weights,
        factorization=factorization,
        init=td_init,
        pretrained_model=model,
    ).cuda()

    n_param = count_parameters(model)
    
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
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) if gamma else None
    if pretrained == "":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=gamma)

    if checkpoint_dir:
        model_state, optimizer_state, scheduler_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)

    train_dataset, valid_dataset, test_dataset = get_f_mnist_loader(data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=data_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch, num_workers=data_workers)

    trainer = Trainer(train_loader, valid_loader, model, optimizer, writer, scheduler=scheduler, save=save)
    
    if pretrained != "":
        train_acc, train_loss = trainer.test(loader="train")
        writer.add_scalar("Accuracy/before_finetuning/train", train_acc)
        writer.add_scalar("Loss/before_finetuning/train", train_loss)
        valid_acc, valid_loss = trainer.test()
        writer.add_scalar("Accuracy/before_finetuning/valid", valid_acc)
        writer.add_scalar("Loss/before_finetuning/valid", valid_loss)


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


def tune_decompose(config, checkpoint_dir, data_dir, *args, **kwargs):
    # print({**kwargs})
    decompose(*args, config=config, checkpoint_dir=checkpoint_dir, data_dir=data_dir, **kwargs)


@app.command()
def hype(
    layers: List[int],
    lr_min: float = 1e-5,
    lr_max: float = 1e0,
    runtype: str ='decompose',
    num_samples: int = 10,
    max_epochs: int = 10,
    gpus_per_trial: int = 1,
    cpus_per_trial: int = 4,
    # layers: List[int] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    pretrained: str = "/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/f_mnist/parn_18_d0.5_256_sgd_l0.1_g0.1/1629473591/cnn_best",
    factorization: str = 'tucker',
    # decompose_weights: bool = True,
    td_init: float = 0.02,
    rank: float = 0.5,
    epochs: int = 200,
    logdir: Path = Path("/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/f_mnist"),
    # freeze_parameters: bool = False,
    batch: int = 256,
    gamma: float = 0,
    model_name: str = "parn",
    seed: int = None,
    data_workers: int = 1,
    data_dir: Path = Path("/bigdata/f_mnist"),
    cuda: str = None,
    cpu: str = None,
    checkpoint_dir: str = None,
) -> None:
    """
    hyperparameter tuning

    input:
        lr: list of [min, max] learning rate
        runtype: string in ['train', 'decompose']
    """
    
    config = {
        "lr": tune.loguniform(lr_min, lr_max),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_epochs,
        grace_period=1,     #TODO: what does grace_period mean?
        reduction_factor=2  #TODO: what does reduction_factor do?
    )

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    if runtype == 'decompose':
        train_func = tune_decompose
    elif runtype == 'train':
        train_func = train
    else:
        raise NotImplementedError

    result = tune.run(
        partial(train_func, 
            layers=layers,
            pretrained=pretrained,
            factorization=factorization,
            # decompose_weights: bool = True,
            td_init=td_init,
            rank=rank,
            epochs=epochs,
            logdir=logdir,
            # freeze_parameters: bool = False,
            batch=batch,
            gamma=gamma,
            model_name=model_name,
            seed=seed,
            data_workers=data_workers,
            data_dir=data_dir,
            cuda=cuda,
            cpu=cpu,
            checkpoint_dir= logdir / "check_dir",
        ),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    app()
