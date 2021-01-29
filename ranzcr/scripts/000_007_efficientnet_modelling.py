#!/usr/bin/env python
# coding: utf-8
"""investigate the effect of data augmentation with adam optimizer without lr scheduler on model performance"""
# NOTE: data augmentation with cosine annealing with warm restart reduced the leaderboard score to 0.914
# TODO: resize shouldn't be part of data augmentation because its is time consuming. So write a multiprocessing script for image resizing
# Investigate data augmentation with Adam optimizer without learning rate scheduling gives any good result
# TODO: add model parameters to Neptune

import os
import sys
import time
import math
import glob
import ast
import random
import neptune
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.transforms.functional as TF
from efficientnet_pytorch import EfficientNet

import cv2
import PIL
from PIL import Image
from PIL import ImageFile

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
from joblib import Parallel, delayed
# In[9]:

## Parameters
IMAGE_SIZE = (224, 224)
IMAGE_SIZE = (512, 512)

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_BACKEND = 'cv2'

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
set_seed()

# In[12]:
def resize_one_image(input_path, output_path, image_size):
    image = cv2.imread(input_path)
    image = cv2.resize(image, image_size)
    cv2.imwrite(output_path, image)

def resize_image_batch(input_dir, output_dir, image_size):
    """multiprocessing image resize function"""
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    input_paths = [os.path.join(input_dir, image_name) for image_name in os.listdir(input_dir)]
    output_paths = [os.path.join(output_dir, image_name) for image_name in os.listdir(input_dir)]
    image_sizes = [image_size]*len(input_paths)
    
    _ = Parallel(n_jobs=-1, verbose=3)(delayed(resize_one_image)(ipath, opath, img_size) for ipath, opath, img_size in zip(input_paths, output_paths, image_sizes))
    


class ImageDataset:
    def __init__(
        self,
        image_paths,
        targets=None,
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
            raise Exception("Invalid combination of "                 "arguments 'grayscale=False' and 'grayscale_as_rgb=True'")
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
        # TODO: add test loader logic
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
            
        if self.targets is not None:
            targets = self.targets[item]
            targets = torch.tensor(targets)
        else: 
            targets = torch.tensor([])
            
        return {
            "image": image,
            "targets": targets,
        }



# path related variable
data_dir = "/home/welcome/github/ranzcr/ranzcr-clip-catheter-line-classification/"
# path checkpoint
path_checkpoints_dir = "/home/welcome/github/ranzcr/checkpoints"
# submission files
path_submissions_dir = "/home/welcome/github/ranzcr/submissions"
# train folds file path
path_train_folds_dir = "/home/welcome/github/ranzcr/train_folds"
# resized image dir
path_resized_train_image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_resized")
print(path_resized_train_image_dir)
# test image resized
path_resized_test_image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_resized")
print(path_resized_test_image_dir)

path_train_dir= os.path.join(data_dir, 'train')
path_test_dir= os.path.join(data_dir, 'test')
path_train_csv= os.path.join(data_dir, 'train.csv')
path_train_annotations= os.path.join(data_dir, 'train_annotations.csv')
path_sample_submission_file= os.path.join(data_dir, 'sample_submission.csv')

# Load data
train_annotations = pd.read_csv(path_train_annotations)
train = pd.read_csv(path_train_csv)
target_cols = [i for i in train.columns if i not in ['StudyInstanceUID', 'PatientID']]

# basic info about data
print(f"number of patients in train csv= {train.PatientID.nunique()}")
print(f"number of study instance id in train csv= {train.StudyInstanceUID.nunique()}")
print(f"number of study instance id train_annotations= {train_annotations.StudyInstanceUID.nunique()}")

# split train and validation set
def create_folds(n_folds=5):
    global train
    # shuffle dataset
    train = train.sample(frac=1, random_state=0).reset_index(drop=True)
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)
    X = train.loc[:, [i for i in train.columns if i not in target_cols]]
    y = train.loc[:, target_cols]

    train.loc[:, 'kfold'] = 0
    for tuple_val in enumerate(mskf.split(X, y)):
        kfold, (train_id, test_idx) = tuple_val
        train.loc[test_idx, 'kfold'] = kfold
    return train


# In[31]:


class DataModule(object):
    def __init__(self, train_dataset, valid_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        
    def get_train_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(self.train_dataset, **kwargs)
    
    def get_valid_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(self.valid_dataset, **kwargs)
    
    def get_test_dataloader(self, **kwargs):
        return torch.utils.data.DataLoader(self.test_dataset, **kwargs)


# In[63]:


class EfficientNetModel(torch.nn.Module):
    def __init__(self, num_labels=11, pretrained=True):
        super().__init__()
        self.num_labels = num_labels
        if pretrained:
            self.backbone = EfficientNet.from_pretrained("efficientnet-b5",)
        else:
            self.backbone = EfficientNet.from_name("efficientnet-b5",)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(2048, num_labels)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, image, targets=None):
        image = self.backbone.extract_features(image)
        image = self.avgpool(image).squeeze(-1).squeeze(-1)
        image = self.dropout(image)
        image = self.fc(image)
        loss = None
        if targets is not None and targets.size(1) == self.num_labels:
            loss = torch.nn.BCEWithLogitsLoss()(image, targets.type_as(image))

        with torch.no_grad():
            y_pred = self.sigmoid(image)

        return y_pred, loss
        


# In[64]:


class Trainer:
    def __init__(self, model, data_module, experiment_id, optimizer=None, scheduler=None, device='cuda'):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.fp16 = False
        self.step_scheduler_after = None
        self.n_epochs = None
        self.metrics = {}
        self.metrics['train'] = []
        self.metrics['valid'] = []
        self.current_epoch = 0
        self.current_batch = 0
        self.scaler = None
        # early stopping related variables
        self._best_score = -np.inf
        self._delta = None
        self._current_score = None
        self._counter = 0
        self._patience = None
        # variable related to model checkpoints
        self.experiment_id = experiment_id
        
    def configure_trainer(self):
        if self.optimizer is None:
            self.configure_optimizers()
        if self.scheduler is None:
            self.configure_schedulers()
        
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
            
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
            
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def configure_schedulers(self, **kwargs):
        if self.optimizer:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, eta_min=3e-6, T_0=2*1004, T_mult=2)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1004*5, eta_min=3e-6)
            # self.scheduler = None
        else:
            raise Exception("optimizer cannot be None type: code fix required")
        
    def set_params(self, **kwargs):
        for parameter, value in kwargs.items():
             setattr(self, parameter, value)

    def compute_metrics(self, batch_outputs_collection, batch_targets_collection):
        y_pred = np.concatenate(batch_outputs_collection, axis=0) 
        y_true = np.concatenate(batch_targets_collection, axis=0) 
        assert y_pred.shape == y_true.shape, "shape mismatch"
        scores = []
        for i in range(y_true.shape[1]):
            try:
                score = roc_auc_score(y_true[:,i], y_pred[:,i])
                scores.append(score)
            except ValueError:
                raise("Behaviour Not expected: FIXME")
        avg_score = np.mean(scores)
        return avg_score
    
    
    def model_forward_pass(self, data):
        """forward pass of model"""
        for key, value in data.items():
            data[key] = value.to(self.device)
            
        if self.fp16:
            with torch.cuda.amp.autocast():
                output, loss = self.model(**data)
        else:
            output, loss = self.model(**data)
                
        return output, loss
        
    
    def train_one_batch(self, data):
        self.optimizer.zero_grad()
        output, loss = self.model_forward_pass(data)
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                if self.step_scheduler_after == "batch":
                    if self.step_scheduler_metric is None:
                        neptune.log_metric('batch_learning_rate', self.scheduler.get_last_lr()[0])
                        self.scheduler.step()
                    else:
                        pass
                        # step_metric = self.metrics['valid']
                        # self.scheduler.step(step_metric)
        neptune.log_metric("train_batch_loss", loss)
        neptune.log_metric("optimizer_batch_lr", self.optimizer.param_groups[0]['lr'])
        if self.scheduler:
            neptune.log_metric("scheduler_batch_lr", self.scheduler.get_last_lr()[0])
        return output, loss
        
    def train_one_epoch(self, dataloader):
        self.model.train()
        all_outputs = []
        all_targets = []
        all_losses = []
        tk0 = tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for batch_id, data in tk0:
            batch_outputs, batch_loss= self.train_one_batch(data)
            all_outputs.append(batch_outputs.detach().cpu().numpy())
            all_targets.append(data['targets'].detach().cpu().numpy())
            all_losses.append(batch_loss.detach().cpu().item())
            tk0.set_postfix(loss=np.array(all_losses).mean(), stage="train", epoch=self.current_epoch)
            # if batch_id == 50:
            #     break
        tk0.close()

        # compute metrics
        # compute average loss
        avg_auc = self.compute_metrics(all_outputs, all_targets)
        avg_loss = np.array(all_losses).mean()
        self.metrics['train'].append({'epoch': self.current_epoch, 
        'avg_loss': avg_loss, 'auc_score': avg_auc})
        neptune.log_metric("train_epoch_loss", avg_loss)
        neptune.log_metric("train_epoch_auc", avg_auc)
        print(self.metrics['train'][self.current_epoch -1])
    
    def validate_one_batch(self, data):
        output, loss = self.model_forward_pass(data)
        neptune.log_metric("valid_batch_loss", loss.detach().cpu())
        return output, loss
 

    def validate_one_epoch(self, dataloader):
        self.model.eval()
        all_outputs = []
        all_targets = []
        all_losses = []
        tk0 = tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for batch_id, data in tk0:
            batch_outputs, batch_loss= self.validate_one_batch(data)
            all_outputs.append(batch_outputs.detach().cpu().numpy())
            all_targets.append(data['targets'].detach().cpu().numpy())
            all_losses.append(batch_loss.detach().cpu().item())
            tk0.set_postfix(loss=np.array(all_losses).mean(), stage="validate", epoch=self.current_epoch)
            # if batch_id == 50:
            #     break
        tk0.close()
        # compute metrics
        # compute average loss
        avg_auc = self.compute_metrics(all_outputs, all_targets)
        avg_loss = np.array(all_losses).mean()
        self.metrics['valid'].append({'epoch': self.current_epoch, 
        'avg_loss': avg_loss, 'auc_score': avg_auc})
        neptune.log_metric("valid_epoch_loss", avg_loss)
        neptune.log_metric("valid_epoch_auc", avg_auc)
        print(self.metrics['valid'][-1])
    
    def early_stoping(self):
        """early stoping function"""
        self._current_score = self.metrics['valid'][-1]['auc_score']
        if (self._current_score - self._best_score) > self._delta:
            self._best_score = self._current_score
            self._counter = 0
            self.save_checkpoint()
            print("early stopping counter reset to 0")
        else:
            self._counter += 1
            print(f"early stopping counter {self._counter} out of {self._patience}")
        if self._counter == self._patience:
            return True
        return False
    
    def save_checkpoint(self):
        """save model and optimizer state for resuming training"""
        if not os.path.isdir(path_checkpoints_dir):
            os.mkdir(path_checkpoints_dir)
        model_path = os.path.join(path_checkpoints_dir, f"{self.experiment_id}.pth")
        print(f"saved the model at {model_path}") 
        model_state_dict = self.model.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        model_dict['lr'] = self.lr
        model_dict['metrics'] = self.metrics
        model_dict['best_score'] = self._best_score
        model_dict['patience'] = self._patience
        model_dict['delta'] = self._delta
        model_dict['train_batch_size'] = self.train_batch_size
        model_dict['validation_batch_size'] = self.validation_batch_size
        model_dict['experiment_id'] = self.experiment_id
        torch.save(model_dict, model_path)
    
    def load(self, model_path, device=None):
        """Load the saved model to resume training and inference"""
        if device:
            self.device = device
        checkpoint = torch.load(model_path)
        if self.model:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.current_epoch = checkpoint['epoch']
        self.fp16 = checkpoint['fp16']
        self.lr = checkpoint['lr']
        self.metrics = checkpoint['metrics']
        self._best_score = checkpoint['best_score']
        self._patience = checkpoint['patience']
        self._delta = checkpoint['delta']
        self.train_batch_size = checkpoint['train_batch_size']
        self.validation_batch_size = checkpoint['validation_batch_size']
        self.experiment_id = checkpoint['experiment_id']
    
    def predict(self, test_batch_size=64, device='cuda', load=False, model_path=None, dataloader_num_workers=4, save_prediction=True):
        """make predictions on test images"""
        self.model.eval()
        self.device = device
        self.test_batch_size = test_batch_size
        if load:
            if model_path:
                self.load(model_path, device=self.device)
            else:
                model_path = os.path.join(path_checkpoints_dir, f"{self.experiment_id}.pth")
                print(f"loaded model={model_path}")
                self.load(model_path, device=self.device)
        if self.model is None:
            raise Exception("model cannot be None. Load or train the model before inference")
        dataloader = self.data_module.get_test_dataloader(batch_size=self.test_batch_size, shuffle=False, num_workers=dataloader_num_workers)
        all_outputs = []
        tk0 = tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for batch_id, data in tk0:
            for key, value in data.items():
                data[key] = value.to(self.device)
            # batch_outputs, batch_loss = self.model(**data)
            batch_outputs, batch_loss= self.validate_one_batch(data)
            all_outputs.append(batch_outputs.detach().cpu().numpy())
        predictions = np.concatenate(all_outputs, axis=0)
        if save_prediction:
            submission = pd.read_csv(path_sample_submission_file)
            assert submission.shape[0] == predictions.shape[0], "unexpected behavior.code fix required"
            submission.iloc[:, 1:] = predictions

            if not os.path.isdir(path_submissions_dir):
                os.mkdir(path_submissions_dir)
            submission.to_csv(os.path.join(path_submissions_dir, f"{self.experiment_id}.csv"), index=False)
        tk0.close()
        return predictions

    def fit(self,
            n_epochs=100, 
            lr=1e-4, 
            step_scheduler_after='epoch', 
            step_scheduler_metric='val_auc',
            device='cuda', 
            fp16=False,
            train_batch_size = 64,
            validation_batch_size=64,
            dataloader_shuffle=True,
            dataloader_num_workers=4,
            tensorboard_writer = None,
            es_delta=1e-4,
            es_patience=5,
           ):
        """fit method to train the model"""
        self.n_epochs = n_epochs
        self.step_scheduler_after = step_scheduler_after
        self.step_scheduler_metric = step_scheduler_metric 
        self.device = device
        self.fp16 = fp16
        self.lr = lr
        self._delta = es_delta
        self._patience = es_patience
        # self.experiment_id = experiment_id
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.configure_trainer()
        # self.set_params(**kwargs)
        for i in range(1, self.n_epochs+1):
            self.current_epoch = i
            # train
            train_dataloader = self.data_module.get_train_dataloader(
                batch_size=train_batch_size, 
                shuffle=dataloader_shuffle, 
                num_workers=dataloader_num_workers,
                pin_memory=True)
            self.train_one_epoch(train_dataloader)
            neptune.log_metric("optimizer_epoch_lr", self.optimizer.param_groups[0]['lr'])
            # validate 
            validation_dataloader = self.data_module.get_valid_dataloader(
                batch_size=validation_batch_size, 
                shuffle=dataloader_shuffle, 
                num_workers=dataloader_num_workers, 
                pin_memory=True
            )
            self.validate_one_epoch(validation_dataloader)
            if self.scheduler:
                if self.step_scheduler_after == 'epoch': 
                    if self.step_scheduler_metric == 'val_auc':
                        neptune.log_metric('scheduler_epoch_lr', self.scheduler.get_last_lr()[0])
                        self.scheduler.step(self.metrics['valid'][-1]['auc_score'])
                    else:
                        neptune.log_metric('scheduler_epoch_lr', self.scheduler.get_last_lr()[0])
                        self.scheduler.step()

            if self.current_epoch > 10 and math.abs(self.scheduler.get_last_lr()[0] - 3e-6) <= 5e-7:
                break
            # es_flag = self.early_stoping()
            # if es_flag:
            #     print(f"early stopping at epoch={i} out of {n_epochs}")
            #     break

def run(fold, resize=False, **kwargs):
    """train single fold classifier"""
    experiment_tag = "000_007_adam_wr"
    experiment_id = f"{experiment_tag}_{fold}"
    # initialize Neptune
    neptune.init(project_qualified_name='aravind/kaggle-ranzcr')
    neptune.create_experiment(f"{experiment_id}")
    neptune.append_tag(experiment_tag)

    if os.path.isfile(os.path.join(path_train_folds_dir, 'train_folds.csv')):
        train = pd.read_csv(os.path.join(path_train_folds_dir, 'train_folds.csv'))
        print("train folds csv read from disk")
    else:
        train = create_folds()
        train.to_csv(os.path.join(path_train_folds_dir, 'train_folds.csv'), index=False)
        print("train folds csv saved to disk for reuse")

    create_folds_count = train.groupby('kfold').StudyInstanceUID.count()
    print(create_folds_count)

    if resize:
        resize_image_batch(path_train_dir, path_resized_train_image_dir, IMAGE_SIZE)

    valid = train.loc[train.kfold == fold].reset_index(drop=True)
    train = train.loc[train.kfold != fold].reset_index(drop=True)

    # image path for torch dataset
    path_train_images = [os.path.join(path_resized_train_image_dir, i + ".jpg") for i in train.StudyInstanceUID.values]
    path_valid_images = [os.path.join(path_resized_train_image_dir, i + ".jpg") for i in valid.StudyInstanceUID.values]
    # test images in the order of submission file
    submission_file = pd.read_csv(path_sample_submission_file)
    path_test_images = [os.path.join(path_resized_test_image_dir, i + ".jpg") for i in submission_file.StudyInstanceUID.values]

    # targets values for torch dataset
    targets_train = train[target_cols].values
    targets_valid = valid[target_cols].values

    print(f"number of train images={len(path_train_images)}")
    print(f"number of validation images={len(path_valid_images)}")
    print(f"train data size={train.shape}")
    print(f"valid data size={valid.shape}")

    train_augmentation = A.Compose([
            A.CLAHE(p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5, rotate_limit=90, scale_limit=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                always_apply=True,
                ),
            ToTensorV2(p=1.0),
            ]
        )


    valid_augmentation = A.Compose([
        A.CLAHE(p=1),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225), 
            max_pixel_value=255.0, 
            always_apply=True,
            ),
        ToTensorV2(),
        ])

    test_augmentation = A.Compose([
        A.CLAHE(p=1),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225), 
            max_pixel_value=255.0, 
            always_apply=True,
            ),
        ToTensorV2(),
        ])

    train_dataset = ImageDataset(
        path_train_images,  
        targets_train,  
        augmentations=train_augmentation,  
        backend=IMAGE_BACKEND,  
        channel_first=True,  
        grayscale=True,  
        grayscale_as_rgb=True,
    )

    valid_dataset = ImageDataset(
        path_valid_images,  
        targets_valid,  
        augmentations=valid_augmentation,  
        backend=IMAGE_BACKEND,  
        channel_first=True,  
        grayscale=True,  
        grayscale_as_rgb=True,
    )

    test_dataset = ImageDataset(
        path_test_images,  
        None,  
        augmentations=test_augmentation,  
        backend=IMAGE_BACKEND,  
        channel_first=True,  
        grayscale=True,  
        grayscale_as_rgb=True,
    )

    model = EfficientNetModel(pretrained=True)
    data_module = DataModule(train_dataset, valid_dataset, test_dataset)
    trainer = Trainer(model, data_module, f"{experiment_id}")
    trainer.fit(**kwargs)

def ensemble_models(model_paths, output_file, model_tag):
    """combine different models to create the ensemble"""
    model = EfficientNetModel(pretrained=False)
    data_module = DataModule(train_dataset, valid_dataset, test_dataset)
    preds_list = []
    num_models = len(model_paths)
    print(f"number of models to ensemble={num_models}")
    for mpath in model_paths:
        if not mpath.split('/')[-1].startswith(model_tag):
            print(f"skipped model={mpath}")
            continue
        else:
            print(f"using model={mpath} for inference")

        trainer = Trainer(model, data_module, None)
        prediction = trainer.predict(load=True, model_path=mpath, save_prediction=False)
        preds_list.append(prediction)

    mean_prediction = np.stack(preds_list, axis=-1).mean(axis=-1)
    print(f"mean prediction array shape={mean_prediction.shape}")
    submission = pd.read_csv(path_sample_submission_file)
    assert submission.shape[0] == mean_prediction.shape[0], "unexpected behavior.code fix required"
    submission.iloc[:, 1:] = mean_prediction
    if not os.path.isdir(path_submissions_dir):
        os.mkdir(path_submissions_dir)

    submission.to_csv(os.path.join(path_submissions_dir, f"{output_file}.csv"), index=False)

if __name__ == '__main__':
    """do some tests here"""
    # trainer.predict(load=True)
    # model_paths = os.listdir(path_checkpoints_dir,)
    # model_paths = [os.path.join(path_checkpoints_dir, mpath) for mpath in model_paths]
    # ensemble_models(model_paths, "000_002_all_folds.csv")
    # print("done")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--resize", required=False, type=bool, default=False)
    args = vars(parser.parse_args())
    fold = args['fold']
    resize = args['resize']
    run(fold, resize, fp16=True, train_batch_size=24, validation_batch_size=8, lr=1e-4, step_scheduler_after='batch', step_scheduler_metric=None)

    
