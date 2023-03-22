import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import locally_linear_embedding as MLLE
from PIL import Image

"""
Initialize Model
"""
DR_components=2
which_modality=[0,1,2,3] # pdw,flair,t2,pre
M=len(which_modality)
data=all[which_modality,middle,:,:]
m,n,k=data.shape

print(f"Running LLE model on slice {middle}...")
vec=data.reshape((M,n*k)).T
#xx2=xx.reshape((M,n,k))
print(np.shape(data))
print(np.shape(vec))

model = MLLE(vec, n_components=DR_components,n_neighbors=70)


"""
Unsupervised Learning

reshape image
"""
proj = model.fit_transform(vec)
print(proj.shape)
proj1=proj.T
data_fused=proj1.reshape((DR_components,n,k))
print(np.shape(data_fused))

"""
Save resulting image
"""
data_fused_resized=[]
data_fused_resized.append (cv2.resize(data_fused[0], [n,k], interpolation = cv2.INTER_LANCZOS4))
print(np.shape(data_fused_resized))    
data_fused_resized.append ( cv2.resize(data_fused[1], [n,k], interpolation = cv2.INTER_LANCZOS4))
print(np.shape(data_fused_resized)) 

plt.imshow(data_fused_resized[0])
