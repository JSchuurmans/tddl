import json
import os
from time import time
from pathlib import Path
from typing import List

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import tltorch

from tddl.data.loaders import get_mnist_loader
from tddl.trainer import Trainer
from tddl.models.cnn import Net, TdNet
from tddl.utils.prime_factors import get_prime_factors
from tddl.utils.random import set_seed
from tddl.models.utils import count_parameters

import typer


app = typer.Typer()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'



@app.command()
def train(
    batch: int = 256,
    epochs: int = 20,
    logdir: Path = Path("./artifacts/mnist"),
    lr: float = 0.001,
    gamma: float = 0.9,
    conv1: int = 32,
    conv2: int = 64,
    seed: int = None,
    data: Path = Path("/bigdata/mnist"),
    cuda: str = None,
):

    if cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda

    if not logdir.is_dir():
        raise FileNotFoundError("{0} folder does not exist!".format(logdir))

    t = round(time())
    
    if seed is None:
        seed = t
    set_seed(seed)

    MODEL_NAME = f"cnn-{conv1}-{conv2}_bn_{batch}_adam_l{lr}_g{gamma}_s{seed == t}"
    logdir = Path(logdir).joinpath(MODEL_NAME,str(t))
    save = {
        "save_every_epoch": None,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": "cnn"
    }

    train_dataset, valid_dataset, test_dataset = get_mnist_loader(data)

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
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))
    model = Net(conv1_out=conv1, conv2_out=conv2).cuda()
    n_param = count_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    trainer = Trainer(train_loader, valid_loader, model, optimizer, writer, scheduler=scheduler, save=save)
    results = trainer.train(epochs=epochs)

    results['n_param'] = n_param
    results['model_name'] = MODEL_NAME
    with open(logdir.joinpath('results.json'), 'w') as f:
        json.dump(results, f)

    writer.close()


@app.command()
def decompose(
    layer_nrs: List[int],
    pretrained: str = "./artifacts/mnist/cnn-32-32_bn_256_adam_l0.01_g0.9/1628155584/cnn_best",
    factorization: str = 'tucker',
    decompose_weights: bool = True,
    td_init: float = 0,
    rank: float = 0.5,
    epochs: int = 10,
    lr: float = 1e-2,
    logdir: str = "./artifacts/mnist",
    freeze_parameters: bool = False,
    batch: int = 256,
    gamma: float = 0.9,
    seed: int = None,
):

    model = torch.load(pretrained)
    n_param_orig = count_parameters(model)
    # fact_model = copy.deepcopy(model)
    
    # which parameters to train
    # if freeze_parameters:
    #     for param in fact_model.parameters():
    #         param.requires_grad = False

    if decompose_weights:
        td_init = False

    # layer_nrs = [2,6] if layer_nr == 0 else [layer_nr]
    decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
    fixed_rank_modes = 'spatial' if factorization == 'tucker' else None

    for i, (name, module) in enumerate(model.named_modules()):
        if i in layer_nrs:
            if type(module) == torch.nn.modules.conv.Conv2d:
                fact_layer = tltorch.FactorizedConv.from_conv(
                    module, 
                    rank=rank, 
                    decompose_weights=decompose_weights, 
                    factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes,
                    decomposition_kwargs=decomposition_kwargs,
                )
            elif type(module) == torch.nn.modules.linear.Linear:
                fact_layer = tltorch.FactorizedLinear.from_linear(
                    module, 
                    in_tensorized_features=get_prime_factors(module.in_features), 
                    out_tensorized_features=get_prime_factors(module.out_features), 
                    rank=rank,
                    factorization=factorization,
                    decomposition_kwargs=decomposition_kwargs,
                )
            if td_init:
                fact_layer.weight.normal_(0, td_init)
            model._modules[name] = fact_layer
    print(model)
    n_param_fact = count_parameters(model)

    t = round(time())
    if seed is None:
        seed = t
    set_seed(seed)
    MODEL_NAME = f"td-{layer_nrs}-{factorization}-{rank}-d{str(decompose_weights)}-i{td_init}_bn_{batch}_adam_l{lr}_g{gamma}_s{seed == t}"
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
        filter(lambda p: p.requires_grad, model.parameters()),
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

    results = trainer.train(epochs=epochs)
    
    results['model_name'] = MODEL_NAME
    results['train_acc_before_ft'] = train_acc
    results['valid_acc_before_ft'] = valid_acc
    results['n_param_fact'] = n_param_fact
    results['n_param_orig'] = n_param_orig
    results['approx_error'] = None # TODO

    with open(logdir.joinpath('results.json'), 'w') as f:
        json.dump(results, f)

    writer.close()


@app.command()
def factorized(
    layer_nrs: List[int],
    factorization: str = 'tucker',
    td_init: float = 0.02,
    rank: float = 0.5,
    epochs: int = 20,
    lr: float = 1e-2,
    logdir: str = "./artifacts/mnist",
    batch: int = 256,
    gamma: float = 0.9,
    conv1: int = 32,
    conv2: int = 32,
    fc1_out: int = 128,
    seed: int = None,
):
    model = TdNet(
        conv1_out=conv1, conv2_out=conv2, fc1_out=fc1_out, 
        layer_nrs=layer_nrs, rank=rank, factorization=factorization, td_init=td_init,
    ).cuda()

    n_param_fact = count_parameters(model)
    print(model)

    t = round(time())
    if seed is None:
        seed = t
    set_seed(seed)
    MODEL_NAME = f"lr-{conv1}-{conv2}-{layer_nrs}-{factorization}-{rank}-i{td_init}_bn_{batch}_adam_l{lr}_g{gamma}_s{seed == t}"
    logdir = Path(logdir).joinpath(MODEL_NAME,str(t))
    save = {
        "save_every_epoch": None,
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
    results = trainer.train(epochs=epochs)

    results['model_name'] = MODEL_NAME
    results['n_param_fact'] = n_param_fact

    with open(logdir.joinpath('results.json'), 'w') as f:
        json.dump(results, f)

    writer.close()


if __name__ == "__main__":
    app()
