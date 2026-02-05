import sys
import inspect
import random
import torch
import copy

from torch.utils.data.dataset import random_split
sys.path.append('src/datasets/')
from cars import Cars
from dtd import DTD
from eurosat import EuroSAT, EuroSATVal
from gtsrb import GTSRB
from mnist import MNIST
from resisc45 import RESISC45
from svhn import SVHN
from sun397 import SUN397
from fgvc import FGVC
from pets import Pets
from flowers import Flowers
from food101 import Food101
from caltech101 import Caltech101
from cifar100 import CIFAR100
from imagenet import ImageNet
from fashionmnist import FashionMNIST
from cifar10 import CIFAR10
from country211 import Country211
from emnist import EMNIST
from fer2013 import CustomFER2013Dataset
from kmnist import KMNIST
from pcam import PCAM
from fer2013 import FER2013
from flowers102 import Flowers102
from oxfordpets import OxfordIIITPet
from sst2 import RenderedSST2
from stl10 import STL10


registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, 
        num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044


    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.test_loader_shuffle = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset



def split_test_into_val_test(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.test_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    test_size = total_size - val_size

    assert val_size > 0
    assert test_size > 0

    lengths = [val_size, test_size]

    valset, testset = random_split(
        dataset.test_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.val_dataset = valset
    new_dataset.val_loader = torch.utils.data.DataLoader(
        new_dataset.val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = testset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.test_loader_shuffle = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset



def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=0, seed=42,
                val_fraction=0.1, max_val_samples=5000, use_val=False, use_small_dataset=False, constrain_batch_size=16):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    
    if use_val:
        dataset = split_test_into_val_test(
            dataset, dataset_name + 'Val', batch_size, num_workers, val_fraction, max_val_samples)
    elif use_small_dataset:
        dataset = split_train_into_train_val(
            dataset, dataset_name + 'Val', batch_size, num_workers, val_fraction, constrain_batch_size)
    return dataset
