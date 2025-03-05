import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np

def pairwise_mahalanobis_distance(X_i,X_j,w):
    # W has shape (rank k<=dim) x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m 

    #Transform poins of X_i,X_j according to W
    if w.dim() == 1:
        #assume cov=0, scale dims by diagonal
        proj_X_i = X_i * w[None,:]
        proj_X_j = X_j * w[None,:]

    else: 
        w = torch.transpose(w,0,1)
        proj_X_i = torch.matmul(X_i,w)
        proj_X_j = torch.matmul(X_j,w)

    return torch.linalg.norm(proj_X_i[:,torch.newaxis,:]  -  proj_X_j[torch.newaxis,:,:],dim=-1)    

def pairwise_mahalanobis_distance_npy(X_i,X_j,w=None):
    # W has shape dim x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m 
    if w is None:
        w = np.identity(X_i.shape[-1])
    else:
        w = w.astype("f")

    X_i = X_i.astype("f")
    X_j = X_j.astype("f")

    #Transform poins of X_i,X_j according to W
    if w.ndim == 1:
        #assume cov=0, scale dims by diagonal
        #w = np.diag(w)
        #proj_X_i = np.matmul(X_i,w)
        #proj_X_j = np.matmul(X_j,w)

        proj_X_i = X_i * w[None,:]
        proj_X_j = X_j * w[None,:]

    else: 
        w = np.transpose(w)
        proj_X_i = np.matmul(X_i,w)
        proj_X_j = np.matmul(X_j,w)

    return np.linalg.norm(proj_X_i[:,np.newaxis,:]  -  proj_X_j[np.newaxis,:,:],axis=-1)  