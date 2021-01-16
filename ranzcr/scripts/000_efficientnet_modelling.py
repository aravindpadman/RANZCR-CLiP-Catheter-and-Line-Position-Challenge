#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
import sys
import time
import math
import glob
import ast
import random
from albumentations.augmentations.transforms import Resize

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.transforms.functional as TF
from efficientnet_pytorch import EfficientNet

from sklearn.metrics import roc_auc_score

import PIL
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3, venn3_circles

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import cv2
import albumentations as A
from torchvision.transforms.transforms import RandomHorizontalFlip
from albumentations.pytorch import ToTensorV2

from PIL import Image
from PIL import ImageFile

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
# %%
writer = SummaryWriter('runs/exp_000') 

# In[9]:


## Parameters
HEIGHT = 224
WIDTH = 224
IMAGE_SIZE = (224, 224)

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_BACKEND = 'pil'

FOLD = 0


# In[10]:


# set seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
set_seed()


# In[11]:


# path related variable
data_dir = "/home/welcome/github/ranzcr/ranzcr-clip-catheter-line-classification/"
# checkpoint directory path
path_checkpoints_dir = "/home/welcome/github/ranzcr/checkpoints"
# submissions directory path
path_submissions_dir = "/home/welcome/github/ranzcr/submissions"


path_train_dir= os.path.join(data_dir, 'train')
path_test_dir= os.path.join(data_dir, 'test')
path_train_csv= os.path.join(data_dir, 'train.csv')
path_train_annotations= os.path.join(data_dir, 'train_annotations.csv')
path_sample_submission_file= os.path.join(data_dir, 'sample_submission.csv')

path_train_images = glob.glob(os.path.join(path_train_dir, "*.jpg"))
path_test_images = glob.glob(os.path.join(path_test_dir, "*.jpg"))

# Load data
train_annotations = pd.read_csv(path_train_annotations)
train = pd.read_csv(path_train_csv)
target_cols = [i for i in train.columns if i not in ['StudyInstanceUID', 'PatientID']]

# basic info about data
print(f"number of patients in train csv= {train.PatientID.nunique()}")
print(f"number of study instance id in train csv= {train.StudyInstanceUID.nunique()}")
print(f"number of study instance id train_annotations= {train_annotations.StudyInstanceUID.nunique()}")


# In[12]:


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
            # targets = torch.tensor([])
            targets = None
            
        return {
            "image": image,
            "targets": targets,
        }


# In[13]:


# image augmentation 
train_augmentation = A.Compose([
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


valid_augmentation = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
    A.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225), 
        max_pixel_value=255.0, 
        always_apply=True,
        ),
    ToTensorV2(),
    ])

test_augmentation = A.Compose([
    A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
    A.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225), 
        max_pixel_value=255.0, 
        always_apply=True,
        ),
    ToTensorV2(),
    ])


# In[14]:


# path related variable
data_dir = "/home/welcome/github/ranzcr/ranzcr-clip-catheter-line-classification/"

path_train_dir= os.path.join(data_dir, 'train')
path_test_dir= os.path.join(data_dir, 'test')
path_train_csv= os.path.join(data_dir, 'train.csv')
path_train_annotations= os.path.join(data_dir, 'train_annotations.csv')
path_sample_submission_file= os.path.join(data_dir, 'sample_submission.csv')

# path_train_images = glob.glob(os.path.join(path_train_dir, "*.jpg"))
# path_test_images = glob.glob(os.path.join(path_test_dir, "*.jpg"))

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

train = create_folds()
create_folds_count = train.groupby('kfold').StudyInstanceUID.count()
print(create_folds_count)


valid = train.loc[train.kfold == FOLD].reset_index(drop=True)
train = train.loc[train.kfold != FOLD].reset_index(drop=True)

# image path for torch dataset
path_train_images = [os.path.join(path_train_dir, i + ".jpg") for i in train.StudyInstanceUID.values]
path_valid_images = [os.path.join(path_train_dir, i + ".jpg") for i in valid.StudyInstanceUID.values]
# test images in the order of submission file
submission_file = pd.read_csv(path_sample_submission_file)
path_test_images = [os.path.join(path_test_dir, i + ".jpg") for i in submission_file.StudyInstanceUID.values]

# targets values for torch dataset
targets_train = train[target_cols].values
targets_valid = valid[target_cols].values

print(f"number of train images={len(path_train_images)}")
print(f"number of validation images={len(path_valid_images)}")

print(f"train data size={train.shape}")
print(f"valid data size={valid.shape}")


# In[15]:


# inititialize dataset and dataloader 

dataset_params = {'batch_size': 64,
                 'shuffle': True,
                 'num_workers': 8,
                 }

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

# train_generator = torch.utils.data.DataLoader(train_dataset, **dataset_params)
# valid_generator = torch.utils.data.DataLoader(valid_dataset, **dataset_params)
# test_generator = torch.utils.data.DataLoader(test_dataset, **dataset_params)


# In[16]:


# for batch in train_generator:
#     print(batch['image'].size(), batch['targets'].size())
#     break
# for batch in valid_generator:
#     print(batch['image'].size(), batch['targets'].size())
#     break
# for batch in test_generator:
#     print(batch['image'].size(), batch['targets'].size())
#     break
# 

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
    def __init__(self, num_labels=11):
        super().__init__()
        self.num_labels = num_labels
        self.backbone = EfficientNet.from_pretrained("efficientnet-b5")
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
        if targets is not None:
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
        self.experiment_id=experiment_id
        
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
                        self.scheduler.step()
                    else:
                        pass
                        # step_metric = self.name_to_metric(self.step_scheduler_metric)
                        # self.scheduler.step(step_metric)
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
        tk0.close()

        # compute metrics
        # compute average loss
        avg_auc = self.compute_metrics(all_outputs, all_targets)
        avg_loss = np.array(all_losses).mean()
        self.metrics['train'].append({'epoch': self.current_epoch, 
        'avg_loss': avg_loss, 'auc_score': avg_auc})
        print(self.metrics['train'][self.current_epoch -1])
    
    def validate_one_batch(self, data):
        output, loss = self.model_forward_pass(data)
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
        # compute metrics
        # compute average loss
        avg_auc = self.compute_metrics(all_outputs, all_targets)
        avg_loss = np.array(all_losses).mean()
        self.metrics['valid'].append({'epoch': self.current_epoch, 
        'avg_loss': avg_loss, 'auc_score': avg_auc})
        print(self.metrics['valid'][self.current_epoch -1])
    
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
        torch.save(model_dict, model_path)
    
    def load(self, model_path, device=None):
        """Load the saved model to resume training and inference"""
        if device:
            self.device = device
        checkpoint = torch.load(model_path)
        if self.model:
            self.model.load_state_dict(checkpoint['state_dict'])
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
    
    def predict(self, test_batch_size=64, device='cuda', load=False, model_path=None, dataloader_num_workers=4):
        """make predictions on test images"""
        self.model.eval()
        self.device = device
        self.test_batch_size = test_batch_size
        if load:
            self.load(model_path, device=self.device)
        if self.model is None:
            raise Exception("model cannot be None. Load or train the model before inference")
        dataloader = self.data_module.get_test_dataloader(batch_size=self.test_batch_size, shuffle=False, num_workers=dataloader_num_workers)
        all_outputs = []
        tk0 = tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for batch_id, data in tk0:
            batch_outputs, batch_loss= self.validate_one_batch(data)
            all_outputs.append(batch_outputs.detach().cpu().numpy())
        predictions = np.concatenate(all_outputs, axis=0)
        submission = pd.read_csv(path_sample_submission_file)
        assert submission.shape[0] == predictions.shape[0], "unexpected behavior.code fix required"
        submission.iloc[:, 1:] = predictions
        if not os.path.isdir(path_submissions_dir):
            os.mkdir(path_submissions_dir)
        submission.to_csv(os.path.join(path_submissions_dir, f"{self.experiment_id}.csv"), index=False)
        tk0.close()


    def fit(self,
            n_epochs=100, 
            lr=1e-3, 
            step_scheduler_after='epoch', 
            device='cuda', 
            fp16=False,
            train_batch_size = 64,
            validation_batch_size=64,
            dataloader_shuffle=True,
            dataloader_num_workers=4,
            tensorboard_writer = None,
            es_delta=1e-4,
            es_patience=3,
           ):
        """fit method to train the model"""
        self.n_epochs = n_epochs
        self.step_scheduler_after = step_scheduler_after
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
        global writer
        for i in range(1, self.n_epochs+1):
            self.current_epoch = i
            # train
            train_dataloader = self.data_module.get_train_dataloader(
                batch_size=train_batch_size, 
                shuffle=dataloader_shuffle, 
                num_workers=dataloader_num_workers,
                pin_memory=True)
            self.train_one_epoch(train_dataloader)
            # validate 
            validation_dataloader = self.data_module.get_valid_dataloader(
                batch_size=validation_batch_size, 
                shuffle=dataloader_shuffle, 
                num_workers=dataloader_num_workers, 
                pin_memory=True
            )
            self.validate_one_epoch(validation_dataloader)
            # add training and validation loss and auc
            writer.add_scalars(
                "train_vs_validation_loss",
                {'train': self.metrics['train'][i-1]['avg_loss'],
                'validation': self.metrics['valid'][i-1]['avg_loss']},
                i,
            )
            writer.add_scalars(
                "train_vs_validation_auc",
                {'train': self.metrics['train'][i-1]['auc_score'],
                'validation': self.metrics['valid'][i-1]['auc_score']},
                i,
            )
            es_flag = self.early_stoping()
            if es_flag:
                print(f"early stopping at epoch={i} out of {n_epochs}")
                break
        writer.close()

if __name__ == '__main__':
    """do some tests here"""
    model = EfficientNetModel()
    data_module = DataModule(train_dataset, valid_dataset, test_dataset)
    trainer = Trainer(model, data_module, f"000_001_{FOLD}")
    # trainer.fit(fp16=True, tensorboard_writer=None)
    trainer.predict(load=True)
