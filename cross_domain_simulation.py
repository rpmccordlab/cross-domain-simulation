# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:51:38 2021

@author: Yang Xu

Simulating cross-domain single-cell mouse brain data
"""

import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

##Scenario 1: Same cellular *composition*, different features------------------
h4c_file = "Zeisel.h5"
adata = sc.read_h5ad(h4c_file)
X = adata.X
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = adata.obs['cell'].values.tolist()
y = np.array(y)

domain1 = X[:,:1200]
domain2 = X[:,800:]

##Scenario 2: One missing cell-type in each domain-----------------------------
h4c_file = "Zeisel.h5"
adata = sc.read_h5ad(h4c_file)
X = adata.X
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = adata.obs['cell'].values.tolist()
y = np.array(y)

domain1 = X[:,:1200]
domain2 = X[:,800:]

##deteling one cell type in each domain
domain1 = domain1[y!="astrocytes_ependymal",:]
y1 = y[y!="astrocytes_ependymal"]
domain2 = domain2[y!="endothelial-mural",:]
y2 = y[y!="endothelial-mural"]

##Scenario 3: Domain 1 is double size of domain 2------------------------------
h4c_file = "Zeisel.h5"
adata = sc.read_h5ad(h4c_file)
X = adata.X
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = adata.obs['cell'].values.tolist()
y = np.array(y)

domain2 = X[:,800:]
y2 = y.copy()

y1 = y.tolist()+y.tolist()
y1 = np.array(y1)
domain1 = np.concatenate((X, X),0)
domain1 = domain1[:,:1200]

##Scenario 4: Uniform cell-type distribution-----------------------------------
h4c_file = "Zeisel.h5"
adata = sc.read_h5ad(h4c_file)
X = adata.X
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = adata.obs['cell'].values.tolist()
y = np.array(y)
celltypes = list(set(y.tolist()))

newX = X[y==celltypes[0],:]
r = np.random.permutation(newX.shape[0])
newX = newX[r,:]
newX = newX[:200,:]
newy = y[y==celltypes[0]]
newy = newy[:200].tolist()
celltypes = list(set(y.tolist()))
for c in celltypes[1:]:
    subX = X[y==c,:]
    r = np.random.permutation(subX.shape[0])
    subX = subX[r,:]
    subX = subX[:200,:]
    newX = np.concatenate((newX,subX),0)
    suby = y[y==c].tolist()[:200]
    newy = newy+suby

newy = np.array(newy)
domain1 = newX[:,:1200]
domain2 = newX[:,800:]

##Scenario 5 Random sampling from each cell-type-------------------------------
h4c_file = "Zeisel.h5"
adata = sc.read_h5ad(h4c_file)
X = adata.X
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = adata.obs['cell'].values.tolist()
y = np.array(y)

r = np.random.permutation(X.shape[0])
domain1 = X[r,:][:2000,:1200]
y1 = y[r][:2000]

r = np.random.permutation(X.shape[0])
domain2 = X[r,:][:2000,800:]
y2 = y[r][:2000]