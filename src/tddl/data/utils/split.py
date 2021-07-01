import os
import shutil
import random
from pathlib import Path

import typer
from tqdm import tqdm


def reservoir_sampling(source: Path, n: int, extension='jpg'):
    """Split data for separate validation or test sets"""

    if type(source) is str:
        source = Path(source)

    pathlist = source.glob(f'*.{extension}')
    rc = []
    for k, path in enumerate(tqdm(pathlist)):
        if k < n:
            rc.append(str(path))
        else:
            i = random.randint(0, k)
            if i < n:
                rc[i] = str(path)

    return rc


def move(rc, src, dest):
    for source in tqdm(rc):
        destination = source.replace(src, dest)
        shutil.move(Path(source), Path(destination))


def main(source, n: int, extension='jpg', src='train', dest='valid'):
    source = Path(source)

    rc = reservoir_sampling(source, n, extension=extension)
    move(rc, src, dest)


if __name__ == "__main__":
    typer.run(main)

    # TODO automate script call
    #   python split.py /bigdata/dogs-vs-cats/train/dog 1000 --dest valid
    #   python split.py /bigdata/dogs-vs-cats/train/cat 1000 --dest valid
    #   python split.py /bigdata/dogs-vs-cats/train/cat 1500 --dest test
    #   python split.py /bigdata/dogs-vs-cats/train/dog 1500 --dest test

    # for dest, n in zip(['valid', 'test'], [1000, 1500]):
    #     for label in ['cat', 'dog']:
