from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.parallel
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from torchvision import datasets, transforms

from tddl.data.sets import DatasetFromSubset


def get_train_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return DataLoader(
        datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


def get_test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


def get_mnist_loader(path):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
    train_dataset, valid_dataset = random_split(dataset, (50000, 10000), generator=torch.Generator().manual_seed(42))
    test_dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
    
    return train_dataset, valid_dataset, test_dataset


def get_f_mnist_loader(path):
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = datasets.FashionMNIST(path, train=True, download=True)
    train_dataset, valid_dataset = random_split(
        dataset,
        (50000, 10000),
        generator=torch.Generator().manual_seed(42),
    )
    train_dataset = DatasetFromSubset(
        train_dataset, transform=transform_train,
    )
    valid_dataset = DatasetFromSubset(
        valid_dataset, transform=transform_test,
    )

    test_dataset = datasets.FashionMNIST(path, train=False, transform=transform_test)
    
    return train_dataset, valid_dataset, test_dataset


def fmnist_stratified_loaders(
    path: Path,
    batch_size: int,
    data_workers: int,
    valid_size: int = 5000,
    random_transform_training: bool = True,
) -> Tuple[DataLoader,...]:
    '''
        input:
        - valid_size: total number of validation observations
    '''
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]) if random_transform_training else transform_test

    dataset = datasets.FashionMNIST(path, train=True, download=True)

    num_train = len(dataset)
    indices = list(range(num_train))

    train_idx, valid_idx, _, _ = train_test_split(indices, 
        dataset.targets, test_size=valid_size, 
        stratify=dataset.targets, random_state=42,
    )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataset = datasets.FashionMNIST(
        root=path, train=True,
        download=True, transform=transform_train,
    )

    valid_dataset = datasets.FashionMNIST(
        root=path, train=True,
        download=True, transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=data_workers,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, 
        sampler=valid_sampler,
        num_workers=data_workers,
    )

    test_dataset = datasets.FashionMNIST(
        path, train=False, transform=transform_test, download=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=data_workers,
    )

    return train_loader, valid_loader, test_loader


def cifar10_stratified_loaders(
    path: Path,
    batch_size: int,
    data_workers: int,
    valid_size: int = 5000,
    random_transform_training: bool = True,
) -> Tuple[DataLoader,...]:
    ...
    '''
        input:
        - valid_size: total number of validation observations
    '''

    mean_CIFAR10 = [0.49139968, 0.48215841, 0.44653091]
    std_CIFAR10 = [0.24703223, 0.24348513, 0.26158784]

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean_CIFAR10,
            std_CIFAR10,
        ),
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean_CIFAR10,
            std_CIFAR10,
        ),
    ]) if random_transform_training else transform_test

    dataset = datasets.CIFAR10(path, train=True, download=True)

    num_train = len(dataset)
    indices = list(range(num_train))

    train_idx, valid_idx, _, _ = train_test_split(indices, 
        dataset.targets, test_size=valid_size, 
        stratify=dataset.targets, random_state=42,
    )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataset = datasets.CIFAR10(
        root=path, train=True,
        download=True, transform=transform_train,
    )

    valid_dataset = datasets.CIFAR10(
        root=path, train=True,
        download=True, transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=data_workers,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, 
        sampler=valid_sampler,
        num_workers=data_workers,
    )

    test_dataset = datasets.CIFAR10(
        path, train=False, transform=transform_test, download=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=data_workers,
    )

    return train_loader, valid_loader, test_loader


def fetch_loaders(
    dataset='fmnist',
    **kwargs,
):
    loaders = {
        'fmnist': fmnist_stratified_loaders,
        'cifar10': cifar10_stratified_loaders,
    }

    return loaders[dataset](**kwargs)
