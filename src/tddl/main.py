from time import time
import copy
from pathlib import Path

import typer
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tddl.models.vgg import ModifiedVGG16Model
from tddl.trainer import Trainer
import tensorly as tl
import tltorch
from torchsummary import summary


app = typer.Typer()

@app.command()
def train(
    epochs: int = 10,
    logdir="/home/jetzeschuurman/gitProjects/phd/tddl/artifacts"
):
    t = round(time())
    logdir = Path(logdir).joinpath(str(t))
    save = {
        "save_every_epoch": 10,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": "model"
    }

    train_path = "/bigdata/dogs-vs-cats/train/"
    valid_path = "/bigdata/dogs-vs-cats/valid/"

    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))

    model = ModifiedVGG16Model().cuda()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)
    trainer = Trainer(train_path, valid_path, model, optimizer, writer, save=save)

    trainer.train(epochs=epochs)

    writer.close()

@app.command()
def decompose(
    pretrained: str = "/home/jetzeschuurman/gitProjects/phd/tddl/artifacts/1625154185/model_52",
    layer_nr: int = 19,
    factorization: str = 'tucker',
    decompose_weights: bool = True,
    layer_type: str = "conv",
    rank: float = 0.5,
    epochs: int = 10,
    lr: float = 1e-3,
    logdir: str = "/home/jetzeschuurman/gitProjects/phd/tddl/artifacts",
    train_path: str = "/bigdata/dogs-vs-cats/train/",
    valid_path: str = "/bigdata/dogs-vs-cats/valid/",
    freeze_parameters: str = 'feat_clas',
):

    model = torch.load(pretrained)
    print(model)
    summary(model, (3, 224, 224))
    fact_model = copy.deepcopy(model)
    # which parameters to train
    if "feat" in freeze_parameters:
        for param in fact_model.features.parameters():
            param.requires_grad = False
    if "clas" in freeze_parameters:
        for param in fact_model.classifier.parameters():
            param.requires_grad = False

    with_init = not decompose_weights

    if layer_type == 'conv':
        conv = model.features[layer_nr]
        fact_conv = tltorch.FactorizedConv.from_conv(conv, rank=rank, decompose_weights=decompose_weights, factorization=factorization)
        if with_init:
            fact_conv.weight.normal_(0,0.02)
        fact_model.features[layer_nr] = fact_conv
        # if update_parameters == 'fact':
        #     for param in fact_model.classifier[layer_nr].parameters():
        #         param.requires_grad = True
    # elif layer_type == 'linear':


    t = round(time())
    logdir = Path(logdir).joinpath(str(t))
    save = {
        "save_every_epoch": 1,
        "save_location": str(logdir),
        "save_best": True,
        "save_final": True,
        "save_model_name": f"fact_model_{layer_type}_{layer_nr}_{factorization}_{lr}_{freeze_parameters}_"+str(with_init)
    }
    writer = SummaryWriter(log_dir=logdir.joinpath('runs'))

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, fact_model.parameters()), 
        lr=lr, 
        momentum=0.99
    )
    trainer = Trainer(train_path, valid_path, fact_model, optimizer, writer, save=save, batch_size=32)

    train_acc = trainer.test(loader="train")
    writer.add_scalar("Accuracy/before_finetuning/train", train_acc)
    valid_acc = trainer.test()
    writer.add_scalar("Accuracy/before_finetuning/valid", valid_acc)

    trainer.train(epochs=epochs)

    writer.close()

if __name__ == "__main__":
    app()
