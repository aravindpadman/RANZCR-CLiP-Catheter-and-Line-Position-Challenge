# %%
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
# %%
# parameters 
HEIGHT = 224
WIDTH = 224
IMAGE_SIZE = (224, 224)

# %%
# cuda setting 
# torch.cuda.empty_cache()

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed()

# %%
data_dir = "/home/welcome/github/RANZCR-CLiP-Catheter-and-Line-Position-Challenge/ranzcr-clip-catheter-line-classification/"

path_train_dir= os.path.join(data_dir, 'train')
path_test_dir= os.path.join(data_dir, 'test')
path_train_csv= os.path.join(data_dir, 'train.csv')
path_train_annotations= os.path.join(data_dir, 'train_annotations.csv')
path_sample_submission_file= os.path.join(data_dir, 'sample_submission.csv')

glob_train_images = glob.glob(os.path.join(path_train_dir, "*.jpg"))
glob_test_images = glob.glob(os.path.join(path_test_dir, "*.jpg"))

# %%
# read tfrec file to check the content
# import tensorflow as tf
#     
# raw_dataset = tf.data.TFRecordDataset("../ranzcr-clip-catheter-line-classification/train_tfrecords/00-1881.tfrec")
# 
# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)
#     break
# %%
# analyze train_annotations.csv file
train_annotations = pd.read_csv(path_train_annotations)
train = pd.read_csv(path_train_csv)
target_cols = [i for i in train.columns if i not in ['StudyInstanceUID', 'PatientID']]

# %%
# image = PIL.Image.open(glob_train_images[0])
# image.show()

def display_image(image_path, fig_size=None, cmap='gray', **kwargs):
    """display image"""
    # FIXME: add annotations and labels in the image
    # FIXME: how to display cv2 image display inline
    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    # fig = plt.figure(figsize=fig_size)
    # img = mpimg.imread(image_path)
    # imgplot = plt.imshow(img, cmap=cmap, **kwargs)
    # plt.show()
    image = PIL.Image.open(image_path)
    return image

# %%
print(f"number of patients in train csv= {train.PatientID.nunique()}")
print(f"number of study instance id in train csv= {train.StudyInstanceUID.nunique()}")

print(f"number of study instance id train_annotations= {train_annotations.StudyInstanceUID.nunique()}")
# %%
# TODO: plot annotations on "'../ranzcr-clip-catheter-line-classification/train/1.2.826.0.1.3680043.8.498.18414377418477975048641307873128330186.jpg'"
# NOTE: ast.literal_eval(train_annotations.iloc[0]['data']) list annotation
# NOTE: argument parser snippet
# import argparse
# import os
# 
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--sub-dir', type=str, default='./subs')
#     args, _ = parser.parse_known_args()
#     return args
# %% 
# BUG: Multiple label detected among NGT and CVC. Take a deeper look
df_ETT = train[(train['ETT - Abnormal'] == 1) | (train['ETT - Borderline'] == 1) | (train['ETT - Normal'] == 1)]
ETT_index = set(df_ETT.index)

df_NGT = train[train[['NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal']].sum(axis=1) >= 1]
NGT_index = set(df_NGT.index)

df_CVC = train[(train[['CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',]].sum(axis=1) >= 1)]
CVC_index = set(df_CVC.index)

df_Swan = train[(train['Swan Ganz Catheter Present'] == 1)]
Swan_index = set(df_Swan.index)

venn2(subsets = (ETT_index.__len__(), NGT_index.__len__(), ETT_index.intersection(NGT_index).__len__()), set_labels = ('ETT', 'NGT'), set_colors=('purple', 'skyblue'), alpha = 0.7);
venn2(subsets = (ETT_index.__len__(), CVC_index.__len__(), ETT_index.intersection(CVC_index).__len__()), set_labels = ('ETT', 'CVC'), set_colors=('purple', 'skyblue'), alpha = 0.7);
venn2(subsets = (CVC_index.__len__(), NGT_index.__len__(), CVC_index.intersection(NGT_index).__len__()), set_labels = ('CVC', 'NGT'), set_colors=('purple', 'skyblue'), alpha = 0.7);
# %%
# NOTE: Class imbalance plot
dt_label_count = train[target_cols].sum(axis=0).sort_values()
sns.barplot(y=dt_label_count.index, x=dt_label_count)
plt.show()
# %%
# NOTE: there is a class imbalance issue, especially for ETT-Abnormal
# TODO: data augmentation, data loader, transfer lerning with effnet backbone, lightning trainer.


# %%
## Create folds here 
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

# TODO: move the below snipet if needed
train = create_folds()
create_folds_count = train.groupby('kfold').StudyInstanceUID.count()
print(create_folds_count)

# %%
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # print(f"index = {index}")
        # Select sample
        study_instance_uid = self.list_IDs[index]
        image_path = os.path.join(path_train_dir, f"{study_instance_uid}.jpg")
        image = PIL.Image.open(image_path).convert('RGB')
        # image = TF.to_tensor(image)
        # image = np.array(image) # *255).astype('uint8')
        # x = torch.from_numpy(self.transform(image=image)["image"]).type(torch.uint8)
        x = self.transform(image)
        
        # x = x.permute(2, 0, 1)
        # x.unsqueeze_(0)
        # print(x.shape)
        # print(type(x))

        # Load data and get label
        y = torch.tensor(
            self.labels.loc[self.labels.StudyInstanceUID == study_instance_uid,
             target_cols].values)
        y.squeeze_(0)

        return x, y



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 8}
max_epochs = 1

FOLD = 0

# image augmentation
# transform_train = A.Compose([ A.Resize(height=HEIGHT, width=WIDTH),
#  A.HorizontalFlip(p=0.5),
# ])

transform_train = transforms.Compose([transforms.Resize(IMAGE_SIZE), 
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),],
            )

transform_val = transforms.Compose([transforms.Resize(IMAGE_SIZE), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),],
            )

transform_test = transforms.Compose([transforms.Resize(IMAGE_SIZE), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),],
            )

# transform_validation = A.Compose([
#     A.Resize(height=HEIGHT, width=WIDTH),
# ])
# 
# transform_test = A.Compose([
#     A.Resize(height=HEIGHT, width=WIDTH),
# ])


partition = dict()
partition['train'] = train.loc[train.kfold != FOLD, 'StudyInstanceUID'].tolist()
partition['validation'] = train.loc[train.kfold == FOLD, 'StudyInstanceUID'].tolist()
labels = train


# Generators
training_set = Dataset(partition['train'], labels, transform_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, transform_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# for local_batch, local_labels in training_generator:
#     # Transfer to GPU
#     print(f"batch size = {local_batch.size()}")
#     print(f"batch label size= {local_labels.size()}")
#     break

# %%
class EfficientNetModel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.backbone = EfficientNet.from_pretrained("efficientnet-b5")
        self.dropout = torch.nn.Dropout(p=0.5)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(2048, num_labels)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x, y=None):
        x = self.backbone.extract_features(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        # x = self.sigmoid(x)
        if y is not None:
            loss = torch.nn.BCEWithLogitsLoss()(x, y.type_as(x))
            y_pred = self.sigmoid(x)
            metrics = {} # self.compute_metrics(y, y_pred)
            return y_pred, loss, metrics
        return self.sigmoid(x), None, {} 


    
model = EfficientNetModel(11)
model = model.to(device)
# out = model(local_batch, local_labels)

# %%
# TODO: 4/01/2021: GOAL: make a submission and secure a leaderbord score
# create training loop (15 min) 
# create validation loop (10 min) 
# build logging mechanism in place (20 mins)
# include progress bar to see the progress (5 mins)
# build inference pipeline (30 mins)
# train the model locally 
# uploade the intermediate dataset to kaggle kernel/or build there
# scheduler will be included in the next iteraion of the modelling
# analyze the model results carefully and decide on the tweaking steps
# add early stoping. need to refer abhishek's notebook and tez
# be agile :)



# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# monitor metrics

def compute_metrics(epoch, batch_outputs, batch_targets):
    # NOTE: moved outside of the model
    y_pred = np.concatenate(batch_outputs, axis=0) 
    y_true = np.concatenate(batch_targets, axis=0) 
    assert y_pred.shape[0] == y_true.shape[0]
    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = roc_auc_score(y_true[:,i], y_pred[:,i])
            scores.append(score)
        except ValueError:
            raise("Behaviour Not expected: FIXME")
    avg_score = np.mean(scores)
    return {'epoch': epoch, 
    'auc': avg_score,
    #'scores': scores,
    }

# training loop
tic = time.time()
for epoch in range(1):  # loop over the dataset multiple times
    print(epoch)

    running_loss = 0.0
    running_metrics = 0.0
    batch_outputs = []
    batch_tagets = []
    for i, (inputs, targets) in enumerate(training_generator, 0):
        print(f"batch={i}")
        inputs, targets = inputs.to(device), targets.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, loss, metrics = model(inputs, targets)
        batch_outputs.append(outputs.detach().cpu().numpy())
        batch_tagets.append(targets.detach().cpu().numpy())
        # get the inputs; data is a list of [inputs, targets]
        with torch.set_grad_enabled(True):
            loss.backward()
            optimizer.step()
            # add scheduler logic here , scheduler after batch

        # print statistics
        running_loss += loss.item()
        if i % 20== 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))

            running_loss = 0.0
    # compute metrics after each epoch
    metrics = compute_metrics(epoch, batch_outputs, batch_tagets)
    print(metrics)

print('Finished Training')
toc = time.time()
print(f"total time for 2 epochs = {toc - tic}")

# %%
# TODO: 4/01/2021
# build validation loop: overfitting/underfitting
# training and validation curve to evaluate the model
# model checkpointig logic: resumable checkpoing
# check label wise ROC and Accuracy to get an idea about perfomance

batch_outputs = []
batch_tagets = []
for i, (inputs, targets) in enumerate(validation_generator, 0):
    if i%10 == 0:
        print(f"batch={i}")
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, loss, metrics = model(inputs, targets)
        batch_outputs.append(outputs.detach().cpu().numpy())
        batch_tagets.append(targets.detach().cpu().numpy())
        # get the inputs; data is a list of [inputs, labels]

# compute metrics after each epoch
metrics = compute_metrics(0, batch_outputs, batch_tagets)
print(metrics)



# %%
