import os
import pickle
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
from dassl.utils import read_json, write_json, check_isfile


@DATASET_REGISTRY.register()
class NICO(DatasetBase):

    dataset_dir = "NICO"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))  # /DATA
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        all_domain, all_class = [], []
        train, val, test = [], [], []
        self.train_data_path1 = os.path.join(self.dataset_dir, 'env_train1.csv')
        self.train_data_path2 = os.path.join(self.dataset_dir, 'env_train2.csv')
        self.val_data_path = os.path.join(self.dataset_dir, 'env_val.csv')
        self.test_data_path = os.path.join(self.dataset_dir, 'env_test.csv')

        for path in [self.train_data_path1, self.train_data_path2, self.val_data_path, self.test_data_path]:
            if os.path.exists(path):
                if 'train' in path:
                    if len(train) == 0:
                        all_domain, all_class, train = self.read_csv(path, all_domain, all_class)
                    else:
                        all_domain, all_class, train_ = self.read_csv(path, all_domain, all_class)
                        # train.extend(train2)
                elif 'val' in path:
                    all_domain, all_class, val = self.read_csv(path, all_domain, all_class)
                else:
                    all_domain, all_class, test = self.read_csv(path, all_domain, all_class)

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset_balance(train+train_, num_shots=num_shots)
        val = self.generate_fewshot_dataset_balance(val, num_shots=64)

        # super().__init__(train_x=train, val=val, test=test, train_samples=train_samples, alldomain=all_domain)
        super().__init__(train_x=train, val=val, test=test, train_samples=val, alldomain=all_domain)

    def read_csv(self, filepath, all_domain, all_class):
        out = []
        with open (filepath, 'r') as f:
            for line in f.readlines():
                if '.' in line and check_isfile(filepath) and os.path.isfile(filepath):
                    attribute = line.split(',')[-1].strip('\n')
                    img_path = line.split(',')[0]
                    domainname = line.split(',')[2].split('_')[1]
                    classname = line.split(',')[-1].strip('\n')
                    self.img_dir = os.path.join(self.dataset_dir, attribute, img_path)
                    if domainname not in all_domain:
                        all_domain.append(domainname)
                    if classname not in all_class:
                        all_class.append(classname)
                    domain_label = all_domain.index(domainname)
                    label = all_class.index(classname)
                    if check_isfile(self.img_dir):
                        item = Datum(impath=self.img_dir, label=int(label), domain=int(domain_label), classname=classname, domainname=domainname)
                        out.append(item)
        return all_domain, all_class, out
                

