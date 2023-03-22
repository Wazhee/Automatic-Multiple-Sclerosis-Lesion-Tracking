import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import locally_linear_embedding as MLLE
from PIL import Image

"""
Load MRI Images 
"""
path='/content/MyDrive/MyDrive/BravermanLab/MS_Project/uthealth_data/31604/anat/'

loc1='pdw_2d_ax.nii.gz'#'dual_2d_ax.nii.gz'
dualx = nib.load(path+loc1)
dual = dualx.get_fdata().T

loc1='t2w_2d_ax.nii.gz'#'dual_2d_ax_e1.nii.gz'
dualx = nib.load(path+loc1)
duale = dualx.get_fdata().T

loc2='flair_2d_ax.nii.gz'
flairx = nib.load(path+loc2)
flair = flairx.get_fdata().T

loc3='t1w_3d_ax.nii.gz'#'t1w_2d_ax.nii.gz'
t1x = nib.load(path+loc3)
t1 = t1x.get_fdata().T

loc4='t1w_2d_ax_pre.nii.gz'
prex = nib.load(path+loc4)
pre = prex.get_fdata().T

loc5='t1w_2d_ax_post.nii.gz'
postx = nib.load(path+loc5)
post = postx.get_fdata().T

print(dual.shape,flair.shape,t1.shape,pre.shape,post.shape)

allx=[];
allx.append(dual)
allx.append(duale)
allx.append(flair)
#allx.append(t1)
allx.append(pre)
allx.append(post)
#all_og=np.array(allx)
#all_og.shape
data_og=allx
len(data_og)


# choose image frame you want to analyze
slice_index=25
middle = slice_index

pdw=dual
t2w=duale
import matplotlib.pyplot as plt

plt.subplot(2,2,1)
plt.imshow(pdw[slice_index],cmap='gray'), plt.title('PDW')
plt.subplot(2,2,2)
plt.imshow(flair[slice_index],cmap='gray'), plt.title('Flair')
plt.subplot(2,2,3)
plt.imshow(t2w[slice_index],cmap='gray'), plt.title('T2W')
plt.subplot(2,2,4)
plt.imshow(post[slice_index],cmap='gray'), plt.title('T1-post')
  
"""
Reduce the shape of the image

- using full image is computationally too expensive (exponential training time)
"""
import cv2

M,n,k=dual.shape
#M=10

scale_percent = 25 # percent of original size
width = int(n * scale_percent / 100)
height = int(k * scale_percent / 100)
dim = (width, height)
print(dim)

imx=[];
for m in range(M):
    resized = cv2.resize(pdw[m], dim, interpolation = cv2.INTER_CUBIC)
    imx.append(resized)

pdw2=np.array(imx)

imx=[];
for m in range(M):
    resized = cv2.resize(t2w[m], dim, interpolation = cv2.INTER_LANCZOS4)
    imx.append(resized)

t2w2=np.array(imx)

imx=[];
for m in range(M):
    resized = cv2.resize(flair[m], dim, interpolation = cv2.INTER_LANCZOS4)
    imx.append(resized)

flair2=np.array(imx)

imx=[];
for m in range(M):
    resized = cv2.resize(t1[m], dim, interpolation = cv2.INTER_LANCZOS4)
    imx.append(resized)

t1w2=np.array(imx)

imx=[];
for m in range(M):
    resized = cv2.resize(pre[m], dim, interpolation = cv2.INTER_LANCZOS4)
    imx.append(resized)

pre2=np.array(imx)

imx=[];
for m in range(M):
    resized = cv2.resize(post[m], dim, interpolation = cv2.INTER_LANCZOS4)
    imx.append(resized)

post2=np.array(imx)

allx=[];
allx.append(pdw2)
allx.append(t2w2)
allx.append(flair2)
#allx.append(t1w2)
allx.append(pre2)
allx.append(post2)
all=np.array(allx)
all.shape
