#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Created on Monday 03.12.2021
@author: Keira - github.com/Keira. Bai
a. Visualize confusion matrix
b. Draw Plot
"""
import os
import PIL
import time
import torch
import glob as gb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # int


# In[ ]:


def DrawPlot(A, B, C, D, Fig_title, ep):
    # plot
    sns.set(font_scale = 2.0)
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.plot(np.arange(0, ep), A)         #  test loss (on epoch end)
    plt.plot(np.arange(0, ep), C)
    plt.title("model loss", fontsize = 25)
    plt.xlabel('epochs', fontsize = 25)
    plt.ylabel('loss', fontsize = 25)
    plt.legend(['train', 'valid'], loc="upper left", fontsize = 25)
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(0, ep), B)  # train accuracy (on epoch end)
    plt.plot(np.arange(0, ep), D)         #  test accuracy (on epoch end)
    plt.title("model scores", fontsize = 25)
    plt.xlabel('epochs', fontsize = 25)
    plt.ylabel('accuracy', fontsize = 25)
    plt.legend(['train', 'valid'], loc="upper left", fontsize = 25)    
    
    plt.savefig(Fig_title, dpi=600,transparent = False)
    plt.close(fig)
    plt.show()


def Drawmatrix(all_y, all_y_pred, label, Fig_title, emo, norm = False):
#  visualizing confusion matrix
    con_mat = confusion_matrix(all_y, all_y_pred, labels = label)
    if norm:#   normlization            
        con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        df=pd.DataFrame(con_mat_norm, index=emo, columns=emo)
    else:
        df=pd.DataFrame(con_mat, index=emo, columns=emo)
    fig = plt.figure(figsize=(15, 15), dpi=100)
    ax = plt.subplot(111)
    sns.set(font_scale = 3.0)
    sns.heatmap(df,annot=True, cmap='Greens',fmt = 'g')
    ax.xaxis.tick_top()
    ax.set_ylabel('True', fontsize = 25)
    ax.set_xlabel('Pred', fontsize = 25)
    ax.tick_params(axis = 'y', labelsize = 25, labelrotation = 45)
    ax.tick_params(axis = 'x', labelsize = 25)
#   saving confusion matrix      
    plt.savefig(Fig_title, dpi=100,transparent = False)
    plt.close(fig)
    plt.show()


def mkdirf(path, filename):
    localtime = time.asctime(time.localtime(time.time()))    
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create new folder")
    Logpath = os.path.join(path, filename)
    file = open(Logpath,'w') 
    file.write('Created on {}, by Keira.Bai\n'.format(localtime)) 
    file.close()       
    return Logpath

def mkdir(path):    
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)     