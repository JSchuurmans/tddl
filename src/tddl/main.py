from time import time
from pathlib import Path

import typer
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tddl.models.vgg import ModifiedVGG16Model
from tddl.trainer import Trainer
import tensorly as tl


def main(
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


if __name__ == "__main__":
    typer.run(main)
