import json
import os
from time import time
from pathlib import Path
from typing import List
from functools import partial

import yaml
import numpy as np
# from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.suggest.bayesopt import BayesOptSearch
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
from tddl.utils.hardware import select_hardware
from tddl.models.utils import count_parameters
from tddl.utils.checks import check_path, check_paths

import tensorly as tl

import typer

app = typer.Typer()


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

    logdir, data_dir = check_paths(logdir, data_dir)

    select_hardware(cuda, cpu)
    if cpu is not None:
        data_workers = int(cpu) # max(int(cpu)-1, 1)
    
    if not logdir.is_dir():
        raise FileNotFoundError("{0} folder does not exist!".format(logdir))
    t = round(time())
    if seed is None:
        seed = t
    set_seed(seed)
    MODEL_NAME = f"{model_name}_{depth}_d{dropout}_{batch}_sgd_l{lr}_g{gamma}_s{seed == t}"
    logdir = logdir.joinpath(str(t), MODEL_NAME)
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
    data_workers: int = 2,
    data_dir: Path = Path("/bigdata/f_mnist"),
    cuda: str = None,
    cpu: str = None,
    checkpoint_dir: str = None,
    config: str = None,
) -> None:

    logdir, data_dir = check_paths(logdir, data_dir)

    select_hardware(cuda, cpu)
    if cpu is not None:
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
        factorization = config['fact']
        rank = config['rank']

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
    
    if seed is None:
        t = round(time())
    else:
        t = seed
    set_seed(seed)

    MODEL_NAME = f"{model_name}-{td}-{layers}-{factorization}-{rank}-d{str(decompose_weights)}-i{td_init}_bn_{batch}_sgd_l{lr}_g{gamma}_s{seed == t}"
    logdir = logdir.joinpath(str(t), MODEL_NAME)
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


# Config needs to be first positional argument of trainable_function for RayTune
def tune_decompose(config, checkpoint_dir, data_dir, *args, **kwargs):
    # print({**kwargs})
    # if config['fact']:
    #     factorization=config['fact']
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
    rank = 0.5,
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
    search_type: str = 'grid',
    metric: str = 'loss',
    mode: str = 'min',
) -> None:
    """
    hyperparameter tuning

    input:
        lr: list of [min, max] learning rate
        runtype: string in ['train', 'decompose']
        search_type: string in ['grid', 'loguniform'] # TODO not implemented yet, will be difficult when more hyperparameters are tuned.
        metric: string in ['loss', 'accuracy']
        mode: string in ['min', 'max']
        rank: float (frational rank) or str (0.25, 0.5, 0.75)
        factorization: all ('tucker', 'cp', 'tt')
    """
    
    logdir, data_dir = check_paths(logdir, data_dir)

    select_hardware(cuda, cpu)
    if cpu is not None:
        data_workers = int(cpu) # max(int(cpu)-1, 1)

    # lr = np.arange(lr_min, lr_max, 1)

    lr = [0.1**x for x in [0,1,2,3,4,5]]
    # np.power(0.1, np.array([0,1,2,3,4,5]), dtype=float)

    
    rank_list = [0.25, 0.5, 0.75] if rank == "all" else [float(rank)]
    fact_list = ['tucker', 'cp', 'tt'] if factorization == "all" else [factorization]
    
    config = {
        # "lr": tune.loguniform(lr_min, lr_max),
        "lr": tune.grid_search(lr), # array([1.e+00, 1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05])
        "rank": tune.grid_search(rank_list),
        "fact": tune.grid_search(fact_list),
    }

    scheduler = ASHAScheduler(
        metric=metric, # "loss"
        mode=mode, # "min"
        max_t=max_epochs,
        grace_period=1,     #TODO: Only stop trials at least this old in time. Unclear what definition of time unit is
        reduction_factor= 2,  #TODO: what does reduction_factor do? Halving, check ASHA paper
    )

    reporter = CLIReporter(
        parameter_columns=["lr", "fact", "rank"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    if runtype == 'decompose':
        train_func = tune_decompose
    elif runtype == 'train':
        train_func = train
    else:
        raise NotImplementedError

    checkpoint_dir = checkpoint_dir if checkpoint_dir else logdir / "check_dir"

    # algo = BayesOptSearch(
    #     metric=metric,
    #     mode=mode,
    #     utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
    #     random_search_steps=30,
    # )
    
    # algo = ConcurrencyLimiter(algo, max_concurrent=10)

    result = tune.run(
        partial(train_func, 
            layers=layers,
            pretrained=pretrained,
            # factorization=config['fact'],
            # decompose_weights: bool = True,
            td_init=td_init,
            # rank=rank,
            epochs=epochs,
            logdir=logdir,
            # freeze_parameters: bool = False,
            batch=batch,
            gamma=gamma,
            model_name=model_name,
            seed=seed,
            data_workers=data_workers,
            data_dir=data_dir,
            # cuda=cuda,
            # cpu=cpu,
            checkpoint_dir= checkpoint_dir,
        ),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        # search_alg=algo,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=logdir / "raytune",
        # name="MEHBLAH", # default: DEFAULT_2021-11-16_13-47-23
        # trial_name_creator=trial_name_string # trial_name_string is function
    )

    best_trial = result.get_best_trial(metric, mode, "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # # Restored previous trial from the given checkpoint
    # tune.run(
    #     "PG",
    #     name="RestoredExp", # The name can be different.
    #     stop={"training_iteration": 10}, # train 5 more iterations than previous
    #     restore="~/ray_results/Original/PG_<xxx>/checkpoint_5/checkpoint-5",
    #     config={"env": "CartPole-v0"},
    # )


@app.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
def main(
    ctx: typer.Context,
    config_path: Path = Path("/home/jetzeschuurman/gitProjects/phd/tddl"),
) -> None:
    
    config_data = {}
    # load yaml / alternative https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python
    config_data = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    
    # typer.echo(f"Config file: {config_data}")
    for item in ctx.args:
        config_data.update([item.strip("--").split('=')])
    # typer.echo(f"Config, modified with inputs: {config_data}")
    
    seed = config_data.get("seed")
    if seed is None:
        t = round(time())
    else:
        t = seed
    config_data.update({"seed":t})

    # write yaml to logdir folder
    config_log = Path(config_data['logdir']) / str(t) / 'config.yml'
    os.makedirs(os.path.dirname(config_log), exist_ok=True)
    with open(config_log, 'w', encoding='utf8') as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False, allow_unicode=True)

    # config_data_log = yaml.load(Path("/home/jetzeschuurman/gitProjects/phd/tddl/tmp/1637934961/config.yml").read_text(), Loader=yaml.Loader)
    
    if config_data.get('tune'):
        hype(**config_data)

        #TODO continue from hyperparameter tuning and train best model.

    elif config_data.get('factorization') is None:
        train(**config_data) # TODO check if function can be called with typer, such that path_string is interpreted as Path instance not str
    else:
        decompose(**config_data)


if __name__ == "__main__":
    app()
