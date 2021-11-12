import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
# from torchvision import datasets, transforms
from tddl.data.sets import DatasetFromSubset


def get_train_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
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
    return data.DataLoader(
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
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, (50000, 10000), generator=torch.Generator().manual_seed(42))
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
    train_dataset, valid_dataset = torch.utils.data.random_split(
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
