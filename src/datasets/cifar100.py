import os
import torch
import torchvision.datasets as datasets


class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=14):
        # Data loading code
        location="../../DataSets/clip_fewshot"
        self.train_dataset = datasets.CIFAR100(
            root=location, train=True, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.CIFAR100(
            root=location, train=False, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )

        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]