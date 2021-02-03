# !/usr/bin/env python
# coding: utf-8
"""investigate the effect of data augmentation with adam optimizer without lr scheduler on model performance"""
# image size has increased to 512x512
# TODO: add resuming logic
# TODO: log model and training parameters to neptune
# TODO: add gradient accumulation if needed
# TODO: add extra tags to neptune to filter experiments fast
# TODO: try out SGD optimizer and weight decay(L2 regularization)
# TODO: log multilabel ROC curve to neptune
# NEXT STEPS: may be in an another scripts 
# experiment with SGD and try to reach an auc of 0.95
# explore techniques to reproduce results in deep learning
# How to combine label annotations

from itertools import accumulate
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

from enum import Enum
# In[9]:

## Parameters
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
    
# set_seed()

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
    def __init__(self, num_labels=11, pretrained=True, backbone: str = None):
        super().__init__()
        self.num_labels = num_labels
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(backbone)
        else:
            self.backbone = EfficientNet.from_name(backbone)
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


class ConfigEnum(Enum):
    """define all parameters here"""
    model_backbone = "efficientnet-b5"
    # dataset and dataloader related
    image_backend = 'cv2'
    device='cuda'
    fp16 = True
    accumulate_grad_steps = 1
    image_size = 512
    seed = 42
    train_batch_size = 24
    valid_batch_size = 8
    test_batch_size = 8
    # optimizer related params
    optimizer_type = "Adam"
    optimizer_params = {'Adam': {
                                'lr': 4e-4, 
                                'betas':(0.9, 0.999), 
                                'eps': 1e-08, 
                                'weight_decay':0, 
                                'amsgrad':False
                                },
                        }
    # scheduler related params
    scheduler_type = "CosineAnnealingWarmRestarts"
    scheduler_params = {'CosineAnnealingWarmRestarts': {
                                                'T_0': 1, 
                                                'T_mult': 2, 
                                                'eta_min': 1e-4, 
                                                'last_epoch': -1, 
                                                'verbose': False
                                            },
                        'CyclicLR': {
                            'base_lr': 1e-8, 
                            'max_lr': 1e-3, 
                            'step_size_up': 10,
                            'step_size_down': None,
                            'cycle_momentum': False,
                                },
                            }
    step_scheduler_after = 'batch'
    step_scheduler_metric = None
    # loss and metric related params
    compute_train_loss_after = 'batch'
    compute_train_metric_after = 'epoch'
    compute_valid_loss_after = 'batch'
    compute_valid_metric_after = 'epoch'
    # training stopping criteria
    training_stopping_criteria = 'SGDR_ensemble'
    stopping_criteria_params = {'early_stopping': {'patience': 5, 'delta': 1e-4},
                                'SGDR_ensemble': {'N': 5, 'M':5,},
                                'max_epoch': {'max_epoch': 10,},
                                }
    max_epoch = 100
    train_dataloader_shuffle = True
    dataloader_num_workers = 5
    # test time augmentation
    TTA = False
    TTA_params = {}
    # train on complete data
    train_on_all_data = True
    # validate after epoch or batch
    validate_after = 'epoch'
    validation_steps = None
    # lr range test
    run_lr_range_test = False

class Trainer:
    def __init__(self, 
    model, 
    data_module, 
    experiment_id=None, 
    experiment_tag=None, 
    image_size=None,
    device=None,
    fp16=None,
    accumulate_grad_steps=None,
    seed=None,
    train_batch_size=None,
    valid_batch_size=None,
    test_batch_size=None,
    dataloader_num_workers=None,
    train_dataloader_shuffle=None,
    optimizer_type=None,
    optimizer_params=None,
    scheduler_type=None, 
    scheduler_params=None,
    step_scheduler_after=None,
    step_scheduler_metric=None,
    compute_train_loss_after=None,
    compute_train_metric_after=None,
    compute_valid_loss_after=None,
    compute_valid_metric_after=None,
    training_stopping_criteria=None,
    stopping_criteria_params=None,
    max_epoch=None,
    train_on_all_data=None,
    validate_after=None,
    validation_steps=None,
    run_lr_range_test=None,
    ):
        self.model = model
        self.data_module = data_module
        self.experiment_id = experiment_id
        self.experiment_tag = experiment_tag

        self.image_size = image_size
        self.device = device
        self.fp16 = fp16
        self.accumulate_grad_steps = accumulate_grad_steps

        self.seed = seed

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_dataloader_shuffle = train_dataloader_shuffle

        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params
        self.optimizer = None

        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params
        self.scheduler = None
        self.step_scheduler_after = step_scheduler_after
        self.step_scheduler_metric = step_scheduler_metric

        self.compute_train_loss_after = compute_train_loss_after
        self.compute_train_metric_after = compute_train_metric_after
        self.compute_valid_loss_after = compute_valid_loss_after
        self.compute_valid_metric_after = compute_valid_metric_after

        self.training_stopping_criteria = training_stopping_criteria
        self.stopping_criteria_params = stopping_criteria_params
        self.max_epoch = max_epoch
        self.train_on_all_data = train_on_all_data
        self.run_lr_range_test = run_lr_range_test

        self._best_score = -np.inf
        self._current_score = None
        self._counter = 0

        self.metrics = {}
        self.metrics['train'] = []
        self.metrics['valid'] = []
        self.current_epoch = 0
        self.current_train_batch = 0
        self.current_valid_batch = 0
        self.scaler = None

        self.num_train_samples = None
        self.num_train_iterations = None

        
        self.validate_after = validate_after
        self.validation_steps = validation_steps

        # SGDR checkpoint count
        self.checkpoint_snapshot = 1
        # configure trainer 
        self.configure_trainer()
        
    def configure_trainer(self):
        """configure trainer here"""
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
            
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.num_train_samples = len(self.data_module.get_train_dataloader())
        self.num_train_iterations = np.ceil(self.num_train_samples/self.train_batch_size)
        print(f"number of train iterations={self.num_train_iterations}")

        if self.step_scheduler_after == 'batch':
            if self.scheduler_type == 'CosineAnnealingWarmRestarts':
                self.scheduler_params['T_0'] = int(self.num_train_iterations*self.scheduler_params['T_0'])
                print(f"scheduler params re-adjusted to {self.scheduler_params}")
            elif self.scheduler_type == "CosineAnnealingLR":
                raise NotImplementedError
            elif self.scheduler_type == 'CyclicLR':
                self.scheduler_params['step_size_up'] = int(self.num_train_iterations*self.scheduler_params['step_size_up'])
                print(f"scheduler params re-adjusted to {self.scheduler_params}")


        self.configure_optimizers()
        self.configure_schedulers()
            
    def configure_optimizers(self):
        """configure optimizer here"""
        if self.optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_params)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), **self.optimizer_params) 
        elif self.optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), **self.optimizer_params) 
        else:                
            self.optimizer = None
    
    def configure_schedulers(self, **kwargs):
        """configure different learning rate scheduler here"""
        if self.scheduler_type == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **self.scheduler_params)
        elif self.scheduler_type == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.scheduler_params)
        elif self.scheduler_type == 'CyclicLR':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, **self.scheduler_params)
        else:
            self.scheduler = None
        
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
        self.model.train()
        self.current_train_batch += 1
        output, loss = self.model_forward_pass(data)
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                if (self.current_train_batch % self.accumulate_grad_steps) == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            if self.scheduler:
                if self.step_scheduler_after == "batch":
                    if self.step_scheduler_metric is None:
                        neptune.log_metric('learning_rate_vs_batch', self.scheduler.get_last_lr()[0])
                        self.scheduler.step()
                    else:
                        pass
                        # step_metric = self.metrics['valid']
                        # self.scheduler.step(step_metric)
        neptune.log_metric("train_batch_loss", loss.detach().cpu())
        if self.run_lr_range_test:
            neptune.log_metric('train_batch_loss_vs_LR',
             self.scheduler.get_last_lr()[0], y=loss.detach().cpu())

        return output, loss
        
    def train_one_epoch(self, dataloader):
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

            if self.validate_after == 'batch' and self.train_on_all_data == False:
                if (self.current_train_batch % self.validation_steps) == 0:
                    validation_dataloader = self.data_module.get_valid_dataloader(
                            batch_size=self.valid_batch_size, 
                            shuffle=self.train_dataloader_shuffle, 
                            num_workers=self.dataloader_num_workers, 
                            pin_memory=True
                        )
                    self.validate_one_epoch(validation_dataloader)
                
        tk0.close()

        # compute metrics
        # compute average loss
        avg_auc = self.compute_metrics(all_outputs, all_targets)
        avg_loss = np.array(all_losses).mean()
        self.metrics['train'].append({'epoch': self.current_epoch, 
        'avg_loss': avg_loss, 'auc_score': avg_auc})
        neptune.log_metric("train_epoch_loss", avg_loss)
        neptune.log_metric("train_epoch_AUC", avg_auc)
        print(self.metrics['train'][self.current_epoch -1])
    
    def validate_one_batch(self, data):
        self.current_valid_batch += 1
        with torch.no_grad():
            output, loss = self.model_forward_pass(data)
        neptune.log_metric("validation_batch_loss", loss.detach().cpu())
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
        tk0.close()
        avg_auc = self.compute_metrics(all_outputs, all_targets)
        avg_loss = np.array(all_losses).mean()
        self.metrics['valid'].append({'epoch': self.current_epoch, 
        'avg_loss': avg_loss, 'auc_score': avg_auc})
        print(self.metrics['valid'][-1])
        if self.run_lr_range_test:
            # loss
            neptune.log_metric('validation_steps_end_loss_vs_LR',
            self.scheduler.get_last_lr()[0], y=avg_loss)
            # AUC
            neptune.log_metric('validation_steps_end_AUC_vs_LR',
            self.scheduler.get_last_lr()[0], y=avg_auc)
            # validation steps end loss and AUC
            neptune.log_metric("validation_steps_end_loss", avg_loss)
            neptune.log_metric("validation_steps_end_AUC", avg_auc)
        else:
            neptune.log_metric("validation_epoch_loss", avg_loss)
            neptune.log_metric("validation_epoch_AUC", avg_auc)
    
    def early_stopping(self):
        """early stopping function"""
        self._current_score = self.metrics['valid'][-1]['auc_score']
        delta = self.stopping_criteria_params['delta']
        patience = self.stopping_criteria_params['patience']
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
    
    def save_checkpoint(self, model_path=None):
        """save model and optimizer state for resuming training"""
        # TODO: include new params based on ConfigEnum
        if not os.path.isdir(path_checkpoints_dir):
            os.mkdir(path_checkpoints_dir)
        if model_path is None:
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
        
        if self.scaler is not None:
            amp_grad_scaler = self.scaler.state_dict()
        else:
            amp_grad_scaler = None

        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict['scaler'] = amp_grad_scaler
        model_dict['image_size'] = self.image_size
        model_dict['device'] = self.device
        model_dict['fp16'] = self.fp16
        model_dict['accumulate_grad_steps'] = self.accumulate_grad_steps

        model_dict['experiment_id'] = self.experiment_id
        model_dict['experiment_tag'] = self.experiment_tag

        model_dict['seed'] = self.seed

        model_dict['train_batch_size'] = self.train_batch_size
        model_dict['valid_batch_size'] = self.valid_batch_size
        model_dict['test_batch_size'] = self.test_batch_size
        model_dict['dataloader_num_workers'] = self.dataloader_num_workers
        model_dict['train_dataloader_shuffle'] = self.train_dataloader_shuffle

        model_dict['optimizer_type'] = self.optimizer_type
        model_dict['optimizer_params'] = self.optimizer_params

        model_dict['scheduler_type'] = self.scheduler_type
        model_dict['scheduler_params'] = self.scheduler_params
        model_dict['step_scheduler_after'] = self.step_scheduler_after
        model_dict['step_scheduler_metric'] = self.step_scheduler_metric

        model_dict['compute_train_loss_after'] = self.compute_train_loss_after
        model_dict['compute_train_metric_after'] = self.compute_train_metric_after
        model_dict['compute_valid_loss_after'] = self.compute_valid_loss_after
        model_dict['compute_valid_metric_after'] = self.compute_valid_metric_after

        model_dict['training_stopping_criteria'] = self.training_stopping_criteria
        model_dict['stopping_criteria_params'] = self.stopping_criteria_params
        model_dict['max_epoch'] = self.max_epoch
        model_dict['train_on_all_data'] = self.train_on_all_data
        model_dict['validate_after'] = self.validate_after
        model_dict['validation_steps'] = self.validation_steps
        model_dict['run_lr_range_test'] = self.run_lr_range_test

        model_dict['_best_score'] = self._best_score
        model_dict['_current_score'] = self._current_score
        model_dict['_counter'] = self._counter

        model_dict['metrics'] = self.metrics
        model_dict['current_epoch'] = self.current_epoch
        model_dict['current_train_batch'] = self.current_train_batch
        model_dict['current_valid_batch'] = self.current_valid_batch

        model_dict['num_train_samples'] = self.num_train_samples
        model_dict['num_train_iterations'] = self.num_train_iterations
        torch.save(model_dict, model_path)
    
    def load(self, model_path):
        """Load the saved model to resume training and inference"""
        # TODO: include new params based on ConfigEnum
        checkpoint = torch.load(model_path)

        self.image_size = checkpoint['image_size']
        self.device = checkpoint['device']
        self.fp16 = checkpoint['fp16']
        self.accumulate_grad_steps = checkpoint['accumulate_grad_steps']
        self.experiment_id = checkpoint['experiment_id']
        self.experiment_tag = checkpoint['experiment_tag']
        self.seed = checkpoint['seed']
        self.train_batch_size = checkpoint['train_batch_size']
        self.valid_batch_size = checkpoint['valid_batch_size']
        self.test_batch_size = checkpoint['test_batch_size']
        self.dataloader_num_workers = checkpoint['dataloader_num_workers']
        self.train_dataloader_shuffle = checkpoint['train_dataloader_shuffle']
        self.optimizer_type = checkpoint['optimizer_type']
        self.optimizer_params = checkpoint['optimizer_params']
        self.scheduler_type = checkpoint['scheduler_type']
        self.scheduler_params = checkpoint['scheduler_params']
        self.step_scheduler_after = checkpoint['step_scheduler_after']
        self.step_scheduler_metric = checkpoint['step_scheduler_metric']
        self.compute_train_loss_after = checkpoint['compute_train_loss_after']
        self.compute_train_metric_after = checkpoint['compute_train_metric_after']
        self.compute_valid_loss_after = checkpoint['compute_valid_loss_after']
        self.compute_valid_metric_after = checkpoint['compute_valid_metric_after']
        self.training_stopping_criteria = checkpoint['training_stopping_criteria']
        self.stopping_criteria_params = checkpoint['stopping_criteria_params']
        self.max_epoch = checkpoint['max_epoch']
        self.train_on_all_data = checkpoint['train_on_all_data']
        self.validate_after = checkpoint['validate_after']
        self.validation_steps = checkpoint['validation_steps']
        self.run_lr_range_test= checkpoint['run_lr_range_test']
        self._best_score = checkpoint['_best_score']
        self._current_score = checkpoint['_current_score']
        self._counter = checkpoint['_counter']
        self.metrics = checkpoint['metrics']
        self.current_epoch = checkpoint['current_epoch']
        self.current_train_batch = checkpoint['current_train_batch']
        self.current_valid_batch = checkpoint['current_valid_batch']
        self.num_train_samples = checkpoint['num_train_samples']
        self.num_train_iterations = checkpoint['num_train_iterations']

        # initialize optimizer, scheduler, and gradient scaler
        self.configure_optimizers()
        self.configure_schedulers()
        
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.model:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)

        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler'])
    
    def stopping_criteria(self):
        """define training stopping criteria here"""
        stop_training = False
        if self.training_stopping_criteria == "early_stopping":
            stop_training = self.early_stopping()
        elif self.training_stopping_criteria == 'SGDR_ensemble':
            T_0 = self.scheduler_params['T_0']
            if self.step_scheduler_after == 'batch':
                T_0 = T_0/self.num_train_iterations
                print(f"sgdr T_0={T_0}")

            T_mult = self.scheduler_params['T_mult']
            N = self.stopping_criteria_params['N']
            M = self.stopping_criteria_params['M']
            possible_epoch = [T_0*math.pow(T_mult, i) for i in range(N)] 
            possible_epoch = np.cumsum(possible_epoch)
            snapshot_epochs = possible_epoch[-M:]
            print(snapshot_epochs)

            if self.current_epoch in snapshot_epochs:
                print("warm restarted learning rate")
                print(f"saving model={self.checkpoint_snapshot} out of {M}")
                model_path = os.path.join(path_checkpoints_dir, 
                f"{self.experiment_tag}_snapshot_{self.checkpoint_snapshot}_epoch_{self.current_epoch}.pth")
                self.checkpoint_snapshot += 1
                self.save_checkpoint(model_path=model_path)

            if self.current_epoch == possible_epoch[-1]:
                stop_training = True

        elif self.training_stopping_criteria == "max_epoch":
            if self.current_epoch == self.stopping_criteria_params['max_epoch']: 
                stop_training = True
                self.save_checkpoint()
            
        return stop_training
            
    
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

    def fit(self):
        """fit method to train the model"""
        for i in range(self.current_epoch, self.max_epoch):
            self.current_epoch += 1
            # train
            train_dataloader = self.data_module.get_train_dataloader(
                    batch_size=self.train_batch_size, 
                    shuffle=self.train_dataloader_shuffle, 
                    num_workers=self.dataloader_num_workers,
                    pin_memory=True
                )
            neptune.log_metric("learning_rate_vs_epoch", self.optimizer.param_groups[0]['lr'])
            self.train_one_epoch(train_dataloader)

            # validate 
            if self.validate_after == 'epoch' and self.train_on_all_data == False and self.run_lr_range_test == False:
                validation_dataloader = self.data_module.get_valid_dataloader(
                        batch_size=self.valid_batch_size, 
                        shuffle=self.train_dataloader_shuffle, 
                        num_workers=self.dataloader_num_workers, 
                        pin_memory=True
                    )
                self.validate_one_epoch(validation_dataloader)

            if self.scheduler:
                if self.step_scheduler_after == 'epoch': 
                    if self.step_scheduler_metric == 'val_auc':
                        self.scheduler.step(self.metrics['valid'][-1]['auc_score'])
                    else:
                        self.scheduler.step()

            if self.run_lr_range_test:
                    neptune.log_metric('validation_epoch_end_AUC_vs_LR', 
                    self.scheduler.get_last_lr()[0], y=self.metrics['valid'][-1]['auc_score'])

            stop_training = self.stopping_criteria()
            if stop_training:
                # backward all the accumulate gradients
                print(f"stoped training at {self.current_epoch} epoch")
                break

    def resume_training(self, model_path=None):
        """resuming training"""
        if model_path is None:
            model_path = os.path.join(path_checkpoints_dir, f"{self.experiment_id}.pth")
        self.load(model_path)
        print(f"resuming training from epoch={self.current_epoch}")
        self.fit()
            
    

def run(fold=None, resize=False):
    """train single fold classifier"""
    experiment_tag = "000_008"
    experiment_id = f"{experiment_tag}_{fold}"
    # parameters
    parameters = {}
    for i in ConfigEnum:
        if i.name == 'optimizer_params':
            parameters['optimizer_params'] = i.value[ConfigEnum.optimizer_type.value]
        elif i.name == 'scheduler_params':
            parameters['scheduler_params'] = i.value[ConfigEnum.scheduler_type.value]
        elif i.name == 'stopping_criteria_params':
            parameters['stopping_criteria_params'] = i.value[ConfigEnum.training_stopping_criteria.value]
        else:
            parameters[i.name] = i.value
    parameters['experiment_id'] = experiment_id
    parameters['experiment_tag'] = experiment_tag
    # initialize Neptune
    neptune.init(project_qualified_name='aravind/kaggle-ranzcr')
    neptune.create_experiment(f"{experiment_id}", params=parameters)
    neptune.append_tag(experiment_tag)
    neptune.append_tag(parameters['optimizer_type'])
    neptune.append_tag(parameters['scheduler_type'])
    neptune.append_tag(parameters['training_stopping_criteria'])

    if os.path.isfile(os.path.join(path_train_folds_dir, 'train_folds.csv')):
        train = pd.read_csv(os.path.join(path_train_folds_dir, 'train_folds.csv'))
        print("train folds csv read from disk")
    else:
        train = create_folds()
        train.to_csv(os.path.join(path_train_folds_dir, 'train_folds.csv'), index=False)
        print("train folds csv saved to disk for reuse")

    if resize:
        resize_image_batch(path_train_dir, path_resized_train_image_dir, IMAGE_SIZE)

    # do not split the data into CV folds if train_on_all_data fold is True
    # create dummy validation data instead
    if ConfigEnum.train_on_all_data.value == True:
        valid = pd.DataFrame(columns=train.columns)
    else:
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
        backend=ConfigEnum.image_backend.value,  
        channel_first=True,  
        grayscale=True,  
        grayscale_as_rgb=True,
    )

    valid_dataset = ImageDataset(
        path_valid_images,  
        targets_valid,  
        augmentations=valid_augmentation,  
        backend=ConfigEnum.image_backend.value,  
        channel_first=True,  
        grayscale=True,  
        grayscale_as_rgb=True,
    )

    test_dataset = ImageDataset(
        path_test_images,  
        None,  
        augmentations=test_augmentation,  
        backend=ConfigEnum.image_backend.value,  
        channel_first=True,  
        grayscale=True,  
        grayscale_as_rgb=True,
    )

    # preproccess and distribute all necessary params to respective modules
    model = EfficientNetModel(pretrained=True, backbone=ConfigEnum.model_backbone.value)
    data_module = DataModule(train_dataset, valid_dataset, test_dataset)
    trainer = Trainer(model, data_module, 
        experiment_id=parameters['experiment_id'],
        experiment_tag=parameters['experiment_tag'],
        image_size=ConfigEnum.image_size.value,
        device=ConfigEnum.device.value,
        fp16=ConfigEnum.fp16.value,
        accumulate_grad_steps=ConfigEnum.accumulate_grad_steps.value,
        seed=ConfigEnum.seed.value,
        train_batch_size=ConfigEnum.train_batch_size.value,
        valid_batch_size=ConfigEnum.valid_batch_size.value,
        test_batch_size=ConfigEnum.test_batch_size.value, 
        dataloader_num_workers=ConfigEnum.dataloader_num_workers.value,
        train_dataloader_shuffle=ConfigEnum.train_dataloader_shuffle.value,
        optimizer_type=ConfigEnum.optimizer_type.value,
        optimizer_params=parameters['optimizer_params'],
        scheduler_type=ConfigEnum.scheduler_type.value, 
        scheduler_params=parameters['scheduler_params'],
        step_scheduler_after=ConfigEnum.step_scheduler_after.value,
        step_scheduler_metric=ConfigEnum.step_scheduler_metric.value,
        compute_train_loss_after=ConfigEnum.compute_train_loss_after.value,
        compute_train_metric_after=ConfigEnum.compute_train_metric_after.value,
        compute_valid_loss_after=ConfigEnum.compute_valid_loss_after.value,
        compute_valid_metric_after=ConfigEnum.compute_valid_metric_after.value,
        training_stopping_criteria=ConfigEnum.training_stopping_criteria.value,
        stopping_criteria_params =parameters['stopping_criteria_params'],
        max_epoch=ConfigEnum.max_epoch.value,
        train_on_all_data=ConfigEnum.train_on_all_data.value,
        validate_after=ConfigEnum.validate_after.value,
        validation_steps=ConfigEnum.validation_steps.value,
        run_lr_range_test=ConfigEnum.run_lr_range_test.value,
    )
    trainer.fit()
    # model_path = "/home/welcome/github/ranzcr/checkpoints/000_008_snapshot_1_epoch_1.pth"
    # trainer.resume_training(model_path)


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
    parser.add_argument("--fold", required=False, type=int, default=None)
    parser.add_argument("--resize", type=bool, required=False, default=False)
    parser.add_argument("--sleep", type=bool, required=False, default=False)
    args = vars(parser.parse_args())
    print(args)
    fold = args['fold']
    resize = args['resize']
    sleep = args['sleep']
    if sleep:
        print("START SLEEP")
        time.sleep(5*60)
    run(fold)

# %%
