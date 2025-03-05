import sys
sys.path.insert(0, '..')
#for local import of parent dict

#Import the usual libraries
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms

from sklearn.metrics.pairwise import pairwise_distances
import scipy

import os
import numpy as np
import scanpy as sc
import pandas as pd

#synth Data
from ggml.generator import get_pointcloud, create_t_triplets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#Optimal Transport
import ot

#Plotting
from ggml.plot import plot_distribution, plot_emb, hier_clustering, plot_ellipses
import matplotlib.pyplot as plt
import seaborn as sns



#Training
lr = 0.02
iterations= 30 #0

#Compuation
n_threads = 64
plot_every_i_iterations = 50
save_every_i_iterations = 6

#GGML
norm = "fro"

neighbor_t = [3] #,5,7
rank_k = [5] #,8433] #,10,25,50,100,200] 
shared_init_over_runs = (len(rank_k) == 1) #for fixed k we can init all param combinations jointly

alphas =  [10**i for i in range(-1,4)]# #,100] #,10,50,100]
lambdas = [10**i for i in range(-1,4)] #[0.1,1,10,100]#,100] #,10,50,100]

print("Alphas:")
print(alphas)

print("Lamdbas:")
print(lambdas)

####
#Dataset
n_cells = 500 #Cause normal dataloader only works for identical sizes
pca_c = None
subsample_patient_ratio=0.6



dataset_folder = "/home/kuehn/ot_metric_learning/damin-ggml/data/datasets/"
'''
#Breastcancer
disease = "breastcancer"
dataset_name = "b8b5be07-061b-4390-af0a-f9ced877a068.h5ad"
label_col="reported_diseases"
patient_col="donor_id"


#Kidney
disease = "kidney"
dataset_name = "1c360b0b-eb2f-45a3-aba9-056026b39fa5.h5ad" #"5ccb0043-bb6f-4f00-b7e1-526d2726de9d.h5ad"
label_col="disease"
patient_col="donor_id"

'''
#Myocardial infarction
dataset_name = "c1f6034b-7973-45e1-85e7-16933d0550bc.h5ad"
disease = "myocard_infarct"

#Myocard on major disease type and patients
#label_col="patient_group"  #Fib,Iz,Myo
#patient_col="donor_id"
#Myocard on specific samples and specific tissue zones
patient_col="sample"
label_col="major_labl"  #BZ,CTRL,FZ,IZ,RZ
'''
'''

pca = None
adata = None

def get_cells_by_patients(adata_path,patient_col="donor_id",label_col="reported_diseases",subsample_patient_ratio=0.25,n_feats=None,filter_genes=False,**kwargs):
    global adata
    adata = sc.read_h5ad(adata_path)

    string_class_labels = np.unique(adata.obs[label_col])

    #detect low variable genes
    if filter_genes:
        gene_var = np.var(adata.X.toarray(),axis=0)

        #filter
        thresh = np.mean(gene_var) #TODO make this not hardcoded and arbitrary
        adata = adata[:,gene_var >thresh]
        print(adata)

    distributions = []
    distributions_class = []
    patient_labels = []
    disease_labels = []
    celltype_node_label = []

    if n_feats is not None:
        global pca
        pca = PCA(n_components=n_feats, svd_solver='auto')
        pca.fit(adata.X)

    unique_patients = np.unique(adata.obs[patient_col])
    unique_patients_subsampled = np.random.choice(unique_patients, size = int(len(unique_patients)*subsample_patient_ratio),replace=False)


    for patient in unique_patients_subsampled:


        patient_adata = adata[adata.obs[patient_col] == patient]

        disease_label = np.unique(patient_adata.obs[label_col].to_numpy())
        string_class_label = disease_label[0]
        if len(disease_label) > 1:
            print("Warning, sample_ids refer to cells with multiple disease labels (likely caused by referencing by patients and having multiple samples from different zones)")

        '''
        #This is only relevant for datasets that contain multiple samples
        unique,pos = np.unique(patient_adata.obs[label_col].to_numpy(),return_inverse=True) #Finds all unique elements and their positions
        for sample in unique:


        counts = np.bincount(pos)                     #Count the number of each unique element
        maxpos = counts.argmax()                      #Finds the positions of the maximum count
        string_class_label = unique[maxpos] #we take the label that occurs most often for this patient
        patient_adata = adata[adata.obs[label_col] == string_class_label ] # and only keep those (this is only relevant for myocardial infarction)
        '''

        if patient_adata.n_obs >= n_cells:
            

            sc.pp.subsample(patient_adata,n_obs = n_cells) 

            #if n_feats == 50:
            #    p_arr = np.asarray(patient_adata.obsm["X_pca"],dtype="f")
            #else:
            p_arr = np.asarray(patient_adata.X.toarray(),dtype="f") #TODO directly sparse scipy matrix to numpy matrix?
            if n_feats is not None:
                p_arr = pca.transform(p_arr)

            distributions.append(p_arr)
            
            disease_labels.append(string_class_label)
            distributions_class.append(np.where(string_class_labels==string_class_label)[0][0])
            patient_labels.append(patient)
            celltype_node_label.append(list(patient_adata.obs['cell_type']))
            

    

    return distributions, distributions_class, patient_labels, disease_labels, celltype_node_label

#distributions, distributions_labels, patient_labels = get_cells_by_patients(dataset_folder+dataset_name,patient_col="donor_id",label_col=label_col)
#print(distributions_labels)

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


# In[3]:


class scRNA_Dataset(Dataset):
    #The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, *args, **kwargs):
        #Generate syntehtic data
        distributions, distributions_labels, patient_labels, disease_labels, celltype_node_labels = get_cells_by_patients(*args, **kwargs)
        
        #Population-level 
        self.distributions = distributions
        self.distributions_labels = distributions_labels
        self.disease_labels = disease_labels
        self.patient_labels = patient_labels

        #Unit-level  TODO
        #self.datapoints = points 
        #self.datapoints_labels = point_labels
        self.celltype_node_labels = celltype_node_labels 

        #Triplets
        self.triplets = create_t_triplets(distributions,distributions_labels,**kwargs) #TODO neighbors as param

    def __len__(self):
        #Datapoints to train are always given as triplets
        return len(self.triplets)

    def __getitem__(self, idx):
        #Returns elements and labels of triplet at idx
        i,j,k = self.triplets[idx]
        return np.stack((self.distributions[i],self.distributions[j],self.distributions[k])),np.stack((self.distributions_labels[i],self.distributions_labels[j],self.distributions_labels[k]))
    
    def compute_OT_on_dists(self,ground_metric = None,w = None,symbols=None):
        D = np.zeros((len(self.distributions),len(self.distributions)))
        for i,distribution_i in enumerate(tqdm(self.distributions)):
            for j,distribution_j in enumerate(self.distributions):
                if i < j:
                    if w is not None:
                        M = pairwise_mahalanobis_distance_npy(distribution_i,distribution_j,w)
                    else:
                        M = sp.spatial.distance.cdist(distribution_i,distribution_j)
                    D[i,j] = ot.emd2([],[],M,numThreads=n_threads)
                else:
                    D[i,j]=D[j,i]
        
        plot_emb(D,method='umap',colors=self.disease_labels,symbols=symbols,legend="Side",title="UMAP",verbose=True,annotation=None,s=200)
        plot_emb(D,method='phate',colors=self.disease_labels,symbols=symbols,legend="Side",title="Phate",verbose=True,annotation=None,s=200)

        hier_clustering(D,self.disease_labels, ax=None,dist_name="W_θ")
        return D



training_data = {}
train_dataset = {}


for t in neighbor_t:
    training_data[t] = scRNA_Dataset(dataset_folder+dataset_name,patient_col=patient_col,label_col=label_col,n_feats=pca_c,filter_genes=True,t=t,subsample_patient_ratio=subsample_patient_ratio)
    train_dataset_t = DataLoader(training_data[t], batch_size=128, shuffle=True)
    for a in alphas:
        for l in lambdas:
                for k in rank_k:
                        
                #test_data = CustomSyntheticDataset(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False)

                        train_dataset[(a,l,t,k)] = train_dataset_t
#test_dataset = DataLoader(test_data, batch_size=64, shuffle=True)


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

#The next row is just to get the feature dimensions from a dict of dataloaders (which we need to start iterating over to get an example datapoint)
dim = next(iter(list(train_dataset.values())[0]))[0].shape[-1] 
dia_only = False



from tqdm import tqdm





def triplet_loss(triplet,w,alpha=torch.scalar_tensor(0.1)):
    X_i,X_j,X_k = triplet

    D_ij = pairwise_mahalanobis_distance(X_i,X_j,w)
    D_jk = pairwise_mahalanobis_distance(X_j,X_k,w)

    W_ij = ot.emd2([],[],M=D_ij,log=False,numThreads=n_threads)
    W_jk = ot.emd2([],[],M=D_jk,log=False,numThreads=n_threads)


    return torch.nn.functional.relu(W_ij -  W_jk + alpha)


# In[8]:


import time

results = {}
thetas = {}
times = {}

#TODO move into loop for different k
#Init Theta according to set rank k
if shared_init_over_runs:
    if k is None:
        k = dim
    if dia_only:
        if k is not None:
            raise Warning    
        w_rand =  torch.distributions.uniform.Uniform(-1,1).sample([dim])
        w_euc = torch.ones((dim))
    else:
        w_rand =  torch.distributions.uniform.Uniform(-1,1).sample([k,dim])   #torch.from_numpy(np.asarray(np.random.uniform(-1,1,(2,2)),dtype="f")) 
        w_euc = torch.diag(torch.ones((dim)))[:k,:] #TODO prob identity func in torch, yes and its .eye and not .identity!

    #TODO check if save to delete (aka how paranoid do i have to be with my tensors)
    w_rand.requires_grad_(requires_grad=True)
    w_rand.retain_grad()
    w_euc.requires_grad_(requires_grad=True) 
    w_euc.retain_grad()

for a,l,t,k in train_dataset.keys():
    print(f"alpha: {a} lambda: {l} rank: {k} neighs:{t}")

    if not shared_init_over_runs:
        if k is None:
            k = dim
        if dia_only:
            if k is not None:
                raise Warning    
            w_rand =  torch.distributions.uniform.Uniform(-1,1).sample([dim])
            w_euc = torch.ones((dim))
        else:
            w_rand =  torch.distributions.uniform.Uniform(-1,1).sample([k,dim])   #torch.from_numpy(np.asarray(np.random.uniform(-1,1,(2,2)),dtype="f")) 
            w_euc = torch.diag(torch.ones((dim)))[:k,:] #TODO prob identity func in torch, yes and its .eye and not .identity!

        #TODO check if save to delete (aka how paranoid do i have to be with my tensors)
        w_rand.requires_grad_(requires_grad=True)
        w_rand.retain_grad()
        w_euc.requires_grad_(requires_grad=True) 
        w_euc.retain_grad()


    if (a,l,t,k) in times.keys():
        continue

    alpha = torch.scalar_tensor(a)
    lambda_ = torch.scalar_tensor(l) 


    w_theta = w_rand.clone()
    w_theta.requires_grad_(requires_grad=True) 
    w_theta.retain_grad()

    #Init loss
    total_loss_random_theta = 0
    total_loss_best_theta = 0

    losses = []
    iteration_losses_total = []

    times[(a,l,t,k)] = []
    
    for i in range(1,iterations+1):
    #Iterations
        optimizer = torch.optim.Adam([w_theta], lr=lr)
        iteration_losses = []

        start_epoch = time.time()
        #print(type(train_dataset))
        #train_dataset_shuffled = random.sample([minibatch for minibatch in train_dataset],k=len(train_dataset))

        for triplets, labels in tqdm(train_dataset[(a,l,t,k)]):
        #Minibatches
            optimizer.zero_grad()
            loss = torch.scalar_tensor(0, requires_grad=True) #TODO doesnt actually require gradient
            for trip,labels in zip(triplets,labels):
            #Triplet

                trip.requires_grad_(requires_grad=True)
                loss = loss + triplet_loss(trip,w_theta,alpha)

                #gradient=w_rand
            #Regularization    
            loss = loss / len(triplets) + lambda_ * torch.linalg.norm(w_theta,ord=norm) #TODO scale one by size of triplets as otherwise batchsize influences weighting of regularization
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
        times[(a,l,t,k)].append(end_epoch)
        
        print(f"Average time for neigh {t} and Rank {k} is {np.average(times[(a,l,t,k)])}") #

        if i%save_every_i_iterations==0:
            np.save(f"/home/kuehn/ot_metric_learning/damin-ggml/data/results/learned_parameters/{disease}/GGML/theta_{a}_{l}_{t}_{k}_iter{i}_L{norm}.npy",w_theta.clone().detach().numpy())


            print(f"saved under: /home/kuehn/ot_metric_learning/damin-ggml/data/results/learned_parameters/{disease}/GGML/theta_{a}_{l}_{t}_{k}_iter{i}_L{norm}.npy")

        if i%plot_every_i_iterations==0 and i>0:
            print(f"Compute all OT distances after {i} iterations")
            D = training_data[(t)].compute_OT_on_dists(w = w_theta.clone().detach().numpy())   

    thetas[(a,l,t,k)]=w_theta.clone().detach().numpy()
    
print("done with trianing")

table = np.zeros((len(rank_k),len(neighbor_t)))

for a,l,t,k in times.keys():
    #print(f"Average time for neigh {t} and Rank {k} is {np.average(times[(a,l,t,k)])}")
    table[rank_k.index(k),neighbor_t.index(t)] = np.average(times[(a,l,t,k)])

print(table)

from ggml.benchmark import knn_from_dists, plot_table, plot_1split
import pandas as pd

table_df = pd.DataFrame(data=table,index=rank_k,columns=neighbor_t).transpose()

print(table_df)
print(table_df.to_latex(index=True,
                #formatters={"name": str.upper},
                float_format="{:.1f}".format,
))
