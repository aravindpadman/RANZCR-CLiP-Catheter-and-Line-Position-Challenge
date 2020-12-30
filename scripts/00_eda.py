# %%
import os
import sys
import time
import math
import glob
import ast
import random

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import PIL
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3, venn3_circles

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



# %%
data_dir = "../ranzcr-clip-catheter-line-classification/"

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

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

# %%

def create_folds(n_folds=5):
    global train
    # shuffle dataset
    train = train.sample(frac=1)
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, random_state=0)
    X = train.loc[:, [i for i in train.columns if i not in target_cols]]
    y = train.loc[:, target_cols]

    train.loc[:, 'kfold'] = 0
    for tuple_val in enumerate(mskf.split(X, y)):
        kfold, train_id, test_idx = tuple_val
        # print(len(train_idx))
        # print(len(test_idx))
        # print(kfold)
        train.loc[test_idx] = kfold

# %%
