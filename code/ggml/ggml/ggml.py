import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms

import ot
import time
from tqdm import tqdm 
import numpy as np

from ggml.distances import pairwise_mahalanobis_distance

def triplet_loss(triplet,w,alpha=torch.scalar_tensor(0.1),n_threads=None):
    X_i,X_j,X_k = triplet

    D_ij = pairwise_mahalanobis_distance(X_i,X_j,w)
    D_jk = pairwise_mahalanobis_distance(X_j,X_k,w)

    W_ij = ot.emd2([],[],M=D_ij,log=False,numThreads=n_threads)
    W_jk = ot.emd2([],[],M=D_jk,log=False,numThreads=n_threads)


    return torch.nn.functional.relu(W_ij -  W_jk + alpha)

def ggml(train_dataset,alpha=1,lambda_=1,rank_k=None,neigh_t=None,dia_only=False,lr=0.01,iterations=30,plot_every_i_iterations=10,full_dataset_for_plotting=None,save_every_i_iterations=10,n_threads=None):
    alpha_torch = torch.scalar_tensor(alpha)
    lambda_torch = torch.scalar_tensor(lambda_) 

    #The next row is just to get the feature dimensions from a dict of dataloaders (which we need to start iterating over to get an example datapoint)
    dim = next(iter(train_dataset))[0].shape[-1] 

    if rank_k is None:
        rank_k = dim

    if dia_only:
        if rank_k is not None:
            raise Warning    
        w_rand =  torch.distributions.uniform.Uniform(-1,1).sample([dim])
        w_euc = torch.ones((dim))
    else:
        w_rand =  torch.distributions.uniform.Uniform(-1,1).sample([rank_k,dim])  
        w_euc = torch.diag(torch.ones((dim)))[:rank_k,:]

    #TODO check if save to delete 
    w_rand.requires_grad_(requires_grad=True)
    w_rand.retain_grad()
    w_euc.requires_grad_(requires_grad=True) 
    w_euc.retain_grad()

    w_theta = w_rand.clone()
    w_theta.requires_grad_(requires_grad=True) 
    w_theta.retain_grad()

    losses = []
    iteration_losses_total = []

    times = []
    
    for i in range(iterations):
    #Iterations
        start_epoch = time.time()
        
        optimizer = torch.optim.Adam([w_theta], lr=lr)
        iteration_losses = []

        #print(type(train_dataset))
        #train_dataset_shuffled = random.sample([minibatch for minibatch in train_dataset],k=len(train_dataset))

        for triplets, labels in tqdm(train_dataset):
        #Minibatches
            optimizer.zero_grad()
            loss = torch.scalar_tensor(0, requires_grad=True) #TODO doesnt actually require gradient
            for trip,labels in zip(triplets,labels):
            #Triplet

                trip.requires_grad_(requires_grad=True)
                loss = loss + triplet_loss(trip,w_theta,alpha_torch,n_threads=n_threads)

                #gradient=w_rand
            #Regularization    
            loss = loss + lambda_torch * torch.linalg.norm(w_theta,ord=1) #TODO scale one by size of triplets as otherwise batchsize influences weighting of regularization
            #loss = loss + torch.linalg.norm(torch.matmul(w_theta.transpose(0,1),w_theta)-torch.eye(dim),ord=1) #penalize derivations from euclidean

            loss.backward()

            iteration_losses.append(loss.clone().detach().numpy()) #triplet_loss(t,w_rand)

            #total_loss_best_theta += triplet_loss(t,w_opt)
            optimizer.step()
            optimizer.zero_grad()
            #w_theta = w_theta - lr * w_theta.grad
            #print(f"Minibatch Loss  {np.average(iteration_losses)}")

            w_theta.grad = None
            w_theta.requires_grad_(requires_grad=True)
            w_theta.retain_grad()
        
        losses = np.concatenate((losses,iteration_losses))
        iteration_losses_total.append(np.sum(iteration_losses))
        print(f"Iteration {i} with Loss  {np.sum(iteration_losses)}")

        end_epoch = time.time() - start_epoch
        times.append(end_epoch)
    

        if i%save_every_i_iterations==0:
            np.save(f"/home/kuehn/ot_metric_learning/damin-ggml/data/results/learned_parameters/myocard_infarct/GGML/theta_{alpha}_{lambda_}_{neigh_t}_{rank_k}.npy",w_theta.clone().detach().numpy())
            print(f"saved under: /home/kuehn/ot_metric_learning/damin-ggml/data/results/learned_parameters/myocard_infarct/GGML/theta_{alpha}_{lambda_}_{neigh_t}_{rank_k}.npy")

        if i%plot_every_i_iterations==0 and i>0 and full_dataset_for_plotting is not None:
            print(f"Compute all OT distances after {i} iterations")
            D = full_dataset_for_plotting.compute_OT_on_dists(w = w_theta.clone().detach().numpy()) 

        return w_theta, times
