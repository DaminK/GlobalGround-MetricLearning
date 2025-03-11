#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '../..')
#for local import of parent dict

import scanpy as sc
import pandas as pd
import numpy as np
import pathlib
import scipy
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

from ggml.benchmark import pivoted_chol
from ggml.plot import plot_heatmap, hier_clustering, plot_emb, plot_ellipses
from ggml.data import get_pointcloud, scRNA_Dataset
from ggml.distances import compute_OT, pairwise_mahalanobis_distance_npy, Computed_Distances
from ggml import ggml
from torch.utils.data import DataLoader
from tqdm import tqdm

import ot

neighbor_t = 3
rank_k = 5 
a = 10
l = 1
lr = 0.02
max_iterations = 5

patient_group_split = True
classify = True
cluster = False
results = {}
results_path = "/home/kuehn/ot_metric_learning/damin-ggml/data/results/classification"

train_size = 0.5
test_size = 0.5
validation_size = 0 #use for hyperparameter tuning

load_scRNA = True
disease = "myocard_infarct"
pca_c = None 
max_cells = 1000 

use_synth = False
n = 100
means = [5,10,15]
offsets = np.arange(0,30,3)
shared_means_x = [0, 40]
shared_means_y = [0, 50] 

n_threads = 64



if use_synth:   
    datasets = {
        "synth_2D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1000,noise_dims=1),
        "synth_200D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1,noise_dims=199)
    }

    train_datasets = {
        "synth_2D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1000,noise_dims=1),
        "synth_200D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1,noise_dims=199)
    }
else:
    datasets = {}
    train_datasets = {}

if load_scRNA:
    #Load scRNA datasets
    dataset_folder = "/home/kuehn/ot_metric_learning/damin-ggml/data/datasets/"
    dataset_loading = {
        "breastcancer":{"path":"b8b5be07-061b-4390-af0a-f9ced877a068","label_col":"reported_diseases","patient_col":"donor_id"},
        "kidney":{"path":"1c360b0b-eb2f-45a3-aba9-056026b39fa5","label_col":"disease","patient_col":"donor_id"},
        "myocard_infarct":{"path":"c1f6034b-7973-45e1-85e7-16933d0550bc","label_col":"patient_group","patient_col":"sample"}, 
    }
    if disease is not None:
        keys = list(dataset_loading.keys())
        for key in keys:
            if key!=disease:
                del dataset_loading[key]

    for dataset_name,loading_info in dataset_loading.items():
        data_dict = {}
        data_dict["dataset"] = scRNA_Dataset(dataset_folder+loading_info["path"],loading_info["patient_col"],loading_info["label_col"],t=neighbor_t,n_feats=pca_c,max_cells = max_cells,filter_genes=True)
        data_dict["dataloader"] = DataLoader(data_dict["dataset"], batch_size=128, shuffle=True)
        data_dict["distributions"],data_dict["distributions_labels"],data_dict["points"], data_dict["point_labels"],  data_dict["patient"] = data_dict["dataset"].get_cells_by_patients()
        datasets[dataset_name] = data_dict

        data_dict2 = {}
        data_dict2["dataset"] = scRNA_Dataset(dataset_folder+loading_info["path"],loading_info["patient_col"],loading_info["label_col"],t=neighbor_t,subsample_patient_ratio=test_size,n_feats=pca_c,max_cells = max_cells,filter_genes=True)
        data_dict2["dataloader"] = DataLoader(data_dict2["dataset"], batch_size=128, shuffle=True)
        data_dict2["distributions"],data_dict2["distributions_labels"],data_dict2["points"], data_dict2["point_labels"],  data_dict2["patient"] = data_dict2["dataset"].get_cells_by_patients()
        train_datasets[dataset_name] = data_dict2

metric_params = {}


from metric_learn import LMNN, LFDA, MLKR, NCA, ITML_Supervised
metrics = {
    "GGML": lambda data: ggml(data["dataloader"],a=a,l=l,k=rank_k,lr=lr,max_iterations=max_iterations,n_threads=n_threads),
    "Euclidean": lambda _: "euclidean",
    "Manhatten": lambda _ : "cityblock",
    "Cosine": lambda _ : "cosine",
    #"LMNN": lambda data:LMNN(n_neighbors=30,random_state=42,learn_rate=1e-6,max_iter=100).fit(data["points"],np.unique(data["point_labels"], return_inverse=True)[1]).get_mahalanobis_matrix(),
    "LFDA": lambda data:LFDA(k=3).fit(data["points"],np.unique(data["point_labels"], return_inverse=True)[1]).get_mahalanobis_matrix(),
    #"NCA" : lambda data: NCA(random_state=42).fit(data["points"],np.unique(data["point_labels"], return_inverse=True)[1]).get_mahalanobis_matrix(),
    "ITML": lambda data: ITML_Supervised(random_state=42).fit(data["points"],np.unique(data["point_labels"], return_inverse=True)[1]).get_mahalanobis_matrix(),
    
}
#Load params for learned metric
metric_params_path = "/home/kuehn/ot_metric_learning/damin-ggml/data/results/learned_parameters"

for d in tqdm(train_datasets):
    print(d)
    if d not in metric_params:
        metric_params[d] = {}
    for m in tqdm(metrics):
        print(m)
        if m not in metric_params[d]:
            try:
                if callable(metrics[m]):
                    #print(datasets[d]["points"])
                    dims = datasets[d]["points"][0].shape[-1]
                    #print(dims)
                    metric_params[d][m]= metrics[m](train_datasets[d])
                else:
                    metric_params[d][m]=np.load(f"{metric_params_path}/{d}/{m}/w_theta.npy")
            except Exception as e: 
                print(e)
                metric_params[d][m] = None


#low rank cholesky approximation (pivoted)
for d in metric_params:
    for m in metric_params[d]:
        if isinstance(metric_params[d][m],np.ndarray):
            print(m)
            print(metric_params[d][m].shape)
            if len(metric_params[d][m])>100:
                get_diag = lambda: np.diagonal(metric_params[d][m]).copy()
                get_row = lambda i: metric_params[d][m][i,:]

                metric_params[d][m] = pivoted_chol(get_diag, get_row, rank_k, err_tol = 1e-6)
                print(metric_params[d][m].shape)



#Precompute ground distances with learned metrics ##actually we are just setting the wrapper instead
precomputed_ground_distances = {}
for d in metric_params:
    print(d)
    precomputed_ground_distances[d]={}
    for m in metric_params[d]:
        print(m)
       
        if metric_params[d][m] is not None:
            precomputed_ground_distances[d][m] = Computed_Distances(np.asarray(datasets[d]["points"],dtype='f'),theta=metric_params[d][m]) 

        else: 
            precomputed_ground_distances[d][m] = None



def datapoint_acc_over_splits(pred,true,index,length):
    prediction = np.zeros((length,2))
    for s_pred,s_true,s_index in zip(pred,true,index):
        prediction[s_index,0] += np.squeeze([s_pred == s_true]) # 1 #correct
        prediction[s_index,1] += np.squeeze([s_pred != s_true]) #+= 1 #false

    accuracy = prediction[:,0] / np.sum(prediction,axis=-1)
    return accuracy





import warnings

from ggml.benchmark import VI, VI_np




from scipy.cluster.hierarchy import dendrogram
import sklearn



#Classification
import sklearn 
import scipy as sp
from ggml.benchmark import knn_from_dists, plot_table, plot_1split
import pickle



for d,data in enumerate(precomputed_ground_distances):
    results[data]={}

    for m, (metric_name, distances) in enumerate(precomputed_ground_distances[data].items()):
        print(f"Data {data} Metric {metric_name}")


        if distances is None:
            results[data][metric_name]=None    
        else:

            try:
                element_level = (distances, datasets[data]["point_labels"])
                distribution_level = (compute_OT(datasets[data]["distributions"],datasets[data]["distributions_labels"],precomputed_distances=distances,numThreads=64),datasets[data]["distributions_labels"])
            except Exception as e:
                print(e)
                results[data][metric_name]=None  

            results[data][metric_name]={"global":{},"ground":{}}
            for l, (distances,labels) in enumerate([element_level,distribution_level]):

                if l==0:
                    neighs = 100
                elif l==1:
                    neighs = 5

                #enforce int labels
                labels = np.unique(labels, return_inverse=True)[1]
    
                
                try:
                    if classify:
                        print("classification")
                        pred, true, score, _ , test_indices= knn_from_dists(distances,labels,method=metric_name,weights="uniform",train_size=train_size,test_size=test_size,n_splits=10,n_neighbors=neighs,distribution_labels=datasets[data]["patient"] if (l==0 and patient_group_split) else None) #'distance' ##, train_indices 
                        result = f"{np.average(score):.2f}±{np.std(score):.2f}"
                        results[data][metric_name]["global" if l==0 else "ground"]["KNN"]=result
                        #datapoints_over_split[f"{data}_{metric_name}"] = datapoint_acc_over_splits(pred,true,index=test_indices,length=len(labels))
                        print(result)
                    else:
                        results[data][metric_name]["global" if l==0 else "ground"]["KNN"]= None
                except Exception as e:
                    results[data][metric_name]=None

                try:
                    if cluster:
                        pred_cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=len(np.unique(labels)) if l==1 else None,distance_threshold= None if l==1 else np.quantile(distances[:,:],0.5),metric='precomputed',linkage='average').fit_predict(distances[:,:])
                        mi_score = sklearn.metrics.mutual_info_score(labels,pred_cluster) #
                        ari_score = sklearn.metrics.adjusted_rand_score(labels,pred_cluster) 
                        vi_score,_,_=VI(pred_cluster,labels,torch=False)
                        print(f"{'global' if l==0 else 'ground'} MI:{mi_score:.2f}  ARI:{ari_score:.2f}   SIL:{vi_score:.2f}")
                        results[data][metric_name]['global' if l==0 else 'ground']["MI"]=mi_score
                        results[data][metric_name]['global' if l==0 else 'ground']["ARI"]=ari_score
                        results[data][metric_name]['global' if l==0 else 'ground']["VI"]=vi_score
                        
                except Exception as e:
                    results[data][metric_name]=None
    print(results)   
    with open(f"{results_path}/{'_'.join(datasets.keys())}_background.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                



method_col = metrics.keys()

results_ground_cols = {(d,s):[] for d in results.keys() for s in next(iter(results[d].values()))["ground"].keys() }
results_global_cols = {(d,s):[] for d in results.keys() for s in next(iter(results[d].values()))["global"].keys() }

for d in results:
    for m in results[d]:
        for s in results[d][m]["ground"]:
            results_ground_cols[(d,s)].append(results[d][m]["ground"][s])
        for s in results[d][m]["global"]:
            results_global_cols[(d,s)].append(results[d][m]["global"][s])    

results_ground_cols["method"] = method_col 
results_global_cols["method"] = method_col 

plot_table(pd.DataFrame(results_ground_cols))
plot_table(pd.DataFrame(results_global_cols))

