import os.path as osp
import os
import random
from copy import deepcopy
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from PIL import Image
import aug_lib

from utils.cutout import SLCutoutPIL

# YOUR_PATH/MyProject/others_prj/query2labels/data/intentonomy
inte_image_path = '/data/sqhy_data/intent_resize'
inte_train_anno_path = '/data/sqhy_data/MMintention/data.json'
inte_val_anno_path = '/data/sqhy_data/MMintention/data.json'
inte_test_anno_path = '/data/sqhy_data/MMintention/data.json'


class InteDataSet(data.Dataset):
    def __init__(self,
                 image_dir=None,
                 input_transform=None,
                 anno_path=None,
                 ):
        self.image_dir = image_dir
        self.input_transform = input_transform
        self.anno_path = anno_path

        self.labels = []
        with open(self.anno_path, 'r') as f:
            self.anno_list = json.load(f)
            print(",,,,,,,,")

    def _load_image(self, index_id):
        return Image.open(index_id).convert("RGB")

    def _get_image_path(self, index):
        modality_label = int(self.anno_list[index]['with_text'])
        img_all_paths = self.anno_list[index]['id']
        # img_all_paths = img_all_paths.split("/")
        img_all_path = os.path.join(self.image_dir, img_all_paths)  # /data_sqhy/MMintention/family_222222
        if modality_label != 0:
            text_path = img_all_path + '/text.txt'
        else:
            text_path = img_all_path
        return text_path

    def __getitem__(self, index):
        text_path = self._get_image_path(index)
        label = self.anno_list[index]['topic']
        return text_path, label

    def __len__(self):
        return len(self.anno_list)


def get_datasets(args):
    trivialaugment = aug_lib.TrivialAugment()
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    test_data_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize])

    if args.dataname == 'MMintention':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = InteDataSet(
            image_dir=args.dataset_dir,
            input_transform=test_data_transform,
            anno_path='/data/sqhy_data/MMintention/data.json',
        )
        val_dataset = InteDataSet(
            image_dir=args.dataset_dir,
            input_transform=test_data_transform,
            anno_path='/data/sqhy_data/MMintention/data.json',
        )
        test_dataset = InteDataSet(
            image_dir=args.dataset_dir,
            input_transform=test_data_transform,
            anno_path='/data/sqhy_data/MMintention/data.json',
        )

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    print("len(test_dataset):", len(test_dataset))
    return train_dataset, val_dataset, test_dataset
