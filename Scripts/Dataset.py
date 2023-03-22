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

def load_images(path):
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
  
  
 if __name__ == "__main__":
  load_images(path)
