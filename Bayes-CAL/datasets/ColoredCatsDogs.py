import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from dassl.utils import read_json, write_json, check_isfile


@DATASET_REGISTRY.register()
class ColoredCatsDogs(DatasetBase):

    dataset_dir = "ColoredCatsDogs"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, cfg.TEST_ENV)

        if os.path.exists(self.split_path):
            train1, train2, val, test = self.read_split(self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        # train_samples = self.generate_train_samples_dataset(train, num_samples=16)
        train1 = self.generate_fewshot_dataset(train1, num_shots=int(num_shots/2))
        train2 = self.generate_fewshot_dataset(train2, num_shots=int(num_shots/2))
        val = self.generate_fewshot_dataset(val, num_shots=16)

        all_domain = ['train1', 'train2', 'test']

        super().__init__(train_x=train1+train2, val=val, test=test, train_samples=test, alldomain=all_domain)

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out1, out2, out3, out4 = [], [], [], []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                if 'train_1' in impath:
                    domain_label = 0
                    domainname = 'train1'
                elif 'train_2' in impath:
                    domain_label = 1
                    domainname = 'train2'
                elif 'val' in impath:
                    domain_label = 2
                    domainname = 'test'
                else:
                    domain_label = 2
                    domainname = 'test'
                if check_isfile(impath):
                    item = Datum(impath=impath, label=int(label), domain=domain_label, classname=classname, domainname=domainname)
                # item = Datum(impath=impath, label=int(label), classname=classname)
                if 'train_1' in impath:
                    out1.append(item)
                elif 'train_2' in impath:
                    out2.append(item)
                elif 'val' in impath:
                    out3.append(item)
                elif 'test' in impath:
                    out4.append(item)
                
            return out1, out2, out3, out4

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train1 = _convert(split["train"])[0]
        train2 = _convert(split["train"])[1]
        val = _convert(split["val"])[2]
        test = _convert(split["test"])[3]

        return train1, train2, val, test



