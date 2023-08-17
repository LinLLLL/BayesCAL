import os
import pickle
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden


@DATASET_REGISTRY.register()
class ColoredMNIST(DatasetBase):

    dataset_dir = "ColoredMNIST"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train1_file = os.path.join(self.dataset_dir, "train1.pt")
        self.train2_file = os.path.join(self.dataset_dir, "train2.pt")
        self.test_file = os.path.join(self.dataset_dir, "test.pt")

        self.train_data_label_tuples = torch.load(self.train2_file) + torch.load(self.train1_file)  # 先0.1后0.2
        self.train2_data_label_tuples = torch.load(self.train2_file)
        self.train1_data_label_tuples = torch.load(self.train1_file)
        self.test_data_label_tuples = torch.load(self.test_file)

        self.transform = None  # transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
        train1 = self.read_data("train1", self.train1_data_label_tuples)
        train2 = self.read_data("train2", self.train2_data_label_tuples)
        test = self.read_data("test", self.test_data_label_tuples)

        # train_x = self.read_data("train_all", self.train_data_label_tuples)

        num_shots = cfg.DATASET.NUM_SHOTS
        train = train1 + train2
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = self.generate_fewshot_dataset(test, num_shots=16)
        # test = [tmp for tmp in test if tmp not in val]
        # all_domain = ['10% green 90% red in digit greater_than_four', 
        # '20% green 80% red in digit greater_than_four', '90% green 10% red in digit greater_than_four']
        all_domain = ['green red digit unbalanced1', 'green red digit unbalanced2', 'green red digit unbalanced3']

        super().__init__(train_x=train, val=val, test=test, train_samples=test, alldomain=all_domain)

    def read_data(self, split_dir, data_label):
        items = []

        for index, data in enumerate(data_label):
            img, label = data
            if '1' in split_dir:
                domain_label = 0
            elif '2' in split_dir:
                domain_label = 1
            else:
                domain_label = 2
            if self.transform:
                img = self.transform(img)
            # img存储在impath
            impath = os.path.join(self.dataset_dir, split_dir, str(index) + '.png')
            if not os.path.exists(impath):
                img.save(impath)

            classname = 'less_than_five' if label == 0 else 'greater_than_four'
            if domain_label == 0:
                domainname = 'green red digit unbalanced1'
            elif domain_label == 1: 
                domainname = 'green red digit unbalanced2'
            else:
                domainname = 'green red digit unbalanced3'
            item = Datum(impath=impath, label=label, domain=domain_label, classname=classname, domainname=domainname)
            items.append(item)

        return items
