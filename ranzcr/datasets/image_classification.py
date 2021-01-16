"""Image classification dataset script"""
import cv2
from numpy.core.fromnumeric import shape
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from typing import List, Optional, Union , Any

__all__ = ['ImageDataset']

class ImageDataset:
    def __init__(
        self,
        image_paths,
        targets,
        augmentations=None,
        backend="pil",
        channel_first=True,
        grayscale=False,
        grayscale_as_rgb=False,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param augmentations: albumentations augmentations
        :param backend: 'pil' or 'cv2'
        :param channel_first: f True Images in (C,H,W) format else (H,W,C) format
        :param grayscale: grayscale flag
        :grayscale_as_rgb: load grayscale images as RGB images for transfer learning purpose
        """
        if grayscale is False and grayscale_as_rgb is True:
            raise Exception("Invalid combination of " \
                "arguments 'grayscale=False' and 'grayscale_as_rgb=True'")
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.backend = backend
        self.channel_first = channel_first
        self.grayscale = grayscale
        self.grayscale_as_rgb = grayscale_as_rgb

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        if self.backend == "pil":
            image = Image.open(self.image_paths[item])
            if self.grayscale is True and self.grayscale_as_rgb is True:
                image = image.convert('RGB')
            image = np.array(image)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        elif self.backend == "cv2":
            if self.grayscale is False or self.grayscale_as_rgb is True: 
                image = cv2.imread(self.image_paths[item])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(self.image_paths[item], cv2.IMREAD_GRAYSCALE)
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
        else:
            raise Exception("Backend not implemented")
        if not isinstance(image, torch.Tensor):
            if self.channel_first is True and image.ndim == 3:
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                image = torch.tensor(image)
        if len(image.size()) == 2:
            image = image.unsqueeze(0)
        return {
            "image": image,
            "targets": torch.tensor(targets),
        }

if __name__ == '__main__':
    # TODO: 
    # write argparser
    # add boilerplate to create img_paths and labels list
    # write both torchvision and albumnetation transformation
    # write the data loader loop
    import os
    import sys
    import glob
    import argparse
    import pandas as pd
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    
    # PARAMS DEFINE BELOW
    IMAGE_SIZE = (224, 224)
    HEIGHT = 224 
    WIDTH = 224
    # 

    def argument_parser():
        arg_parser = argparse.ArgumentParser(description="Test image dataset script")
        arg_parser.add_argument('--backend', action='store', help='select "pil" or "cv2"')
        arg_parser.add_argument('--channel_first', action='store_true', help='(*, C, H, W) or (*, H, W, C) format')
        arg_parser.add_argument('--grayscale', action='store_false', help='false for color else True')
        arg_parser.add_argument('--grayscale_as_rgb', action='store_false', help='load grayscale image as RGB image')
        return arg_parser.parse_args()
    # args = argument_parser()
    # params = vars(args)
    # print(params)

    # load and format data for dataloder 
    data_dir = "/home/welcome/github/" \
    "RANZCR-CLiP-Catheter-and-Line-Position-Challenge/ranzcr-clip-catheter-line-classification/"

    path_train_dir= os.path.join(data_dir, 'train')
    path_train_csv= os.path.join(data_dir, 'train.csv')
    path_train_annotations= os.path.join(data_dir, 'train_annotations.csv')
    path_sample_submission_file= os.path.join(data_dir, 'sample_submission.csv')

    train_images_paths = glob.glob(os.path.join(path_train_dir, "*.jpg"))
    train = pd.read_csv(path_train_csv)
    target_cols = [i for i in train.columns if i not in ['StudyInstanceUID', 'PatientID']]
    # image augmentation 
    transform_albumnt = A.Compose([
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225), 
            max_pixel_value=255.0, 
            always_apply=True,
            ),
        ToTensorV2(),
        ])

    #  dataset params
    dataset_params = {'batch_size': 64, 'shuffle': True, 'num_workers': 8}
    max_epochs = 1
    # dataset and dataloder
    image_paths = [os.path.join(path_train_dir, study_id + ".jpg") 
        for study_id in train.StudyInstanceUID.values]
    targets = train[target_cols].values
    # set arguments
    augmentations = transform_albumnt
    backend = 'cv2'
    channel_first = True
    grayscale = True
    grayscale_as_rgb = True

    dataset = ImageDataset(image_paths, 
            targets, 
            augmentations=augmentations, 
            backend=backend, 
            channel_first=channel_first, 
            grayscale=grayscale, 
            grayscale_as_rgb=grayscale_as_rgb,
        )
    training_generator = torch.utils.data.DataLoader(dataset, **dataset_params)
    res = next(iter(training_generator))
    print(res['image'].size())
    print(res['targets'].size())