import os
import shutil
from pathlib import Path

import typer
from tqdm import tqdm


def move_classes(path):
    labels = ['cat', 'dog']

    path = Path(path)

    for label in labels:
        target = path.joinpath(label)
        if not os.path.exists(target):
            os.mkdir(target)

    for filename in tqdm(os.listdir(path)):
        split = filename.split(".")
        label = split[0]
        name = ".".join(split[1:])
        source = path.joinpath(filename)
        destination = path.joinpath(label, name)
        shutil.move(source, destination)





if __name__ == "__main__":
    typer.run(process_data)
