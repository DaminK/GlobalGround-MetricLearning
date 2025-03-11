

import sys
sys.path.insert(0, '..')
#for local import of parent dict

import scanpy as sc
import pandas as pd
import numpy as np
import pathlib
import scipy
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

from ggml.plot import plot_heatmap, hier_clustering, plot_emb, plot_ellipses #imports pydiffmap
from ggml.generator import get_pointcloud
from tqdm import tqdm

import ot

from metric_learn import LMNN, LFDA, MLKR, NCA, ITML_Supervised


# In[2]:


#move into package TODO

def get_cells_by_patients(adata_path,patient_col="donor_id",label_col="reported_diseases",subsample_patient_ratio=0.5,n_feats=None,max_cells = None,filter_genes=False,**kwargs):
    global adata
    adata = sc.read_h5ad(adata_path+".h5ad")  #adata_path+".h5ad")
    

    string_class_labels = np.unique(adata.obs[label_col])

    if filter_genes:
        #detect low variable genes
        gene_var = np.var(adata.X.toarray(),axis=0)
        #filter
        thresh = np.mean(gene_var) #TODO make this not hardcoded and arbitrary
        adata = adata[:,gene_var >thresh]
        #adata.write(adata_path+"_filtered.h5ad")
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
        unique,pos = np.unique(patient_adata.obs[label_col].to_numpy(),return_inverse=True) #Finds all unique elements and their positions
        counts = np.bincount(pos)                     #Count the number of each unique element
        maxpos = counts.argmax()                      #Finds the positions of the maximum count
        string_class_label = unique[maxpos] #we take the label that occurs most often for this patient
        patient_adata = adata[adata.obs[label_col] == string_class_label ] # and only keep those (this is only relevant for myocardial infarction)
        '''

        if max_cells is not None and patient_adata.n_obs > max_cells:
            sc.pp.subsample(patient_adata,n_obs = max_cells) 

        #p_arr = np.asarray(patient_adata.X.toarray(),dtype="f") #TODO directly sparse scipy matrix to numpy matrix?

        #if n_feats == 50:
        #    p_arr = np.asarray(patient_adata.obsm["X_pca"],dtype="f")
        #else:
        p_arr = np.asarray(patient_adata.X.toarray(),dtype="f") #TODO directly sparse scipy matrix to numpy matrix?
        if n_feats is not None:
            p_arr = pca.transform(p_arr)

        distributions.append(p_arr)
        
        disease_labels.append(string_class_label)
        #distributions_class.append(np.where(string_class_labels==string_class_label)[0][0])
        patient_labels.append(list(patient_adata.obs[patient_col]))
        celltype_node_label.append(list(patient_adata.obs['cell_type'])) #cell_Type
            
        #Cell level
    points = np.concatenate(distributions) #np.reshape(np.asarray(dists),(-1,2))
    point_labels = sum([[l] * len(D) for l,D in zip(disease_labels,distributions)],[]) #flattens list of lists

    #return distributions, distributions_class, patient_labels, disease_labels, celltype_node_label
    return distributions, disease_labels, points, point_labels, celltype_node_label, np.concatenate(patient_labels)

    distributions, distributions_class, patient_labels, disease_labels, 

#distributions, distributions_labels, patient_labels = get_cells_by_patients(dataset_folder+dataset_name,patient_col="donor_id",label_col=label_col)
#print(distributions_labels)

def compute_OT(distributions,labels,precomputed_distances=None,ground_metric = None,w = None,legend=None,numThreads=32):
    D = np.zeros((len(distributions),len(distributions)))
    for i,distribution_i in enumerate(distributions):
        for j,distribution_j in enumerate(distributions):
            if i < j:
                if precomputed_distances is not None:
                    start_i = int(np.sum([len(dist) for dist in distributions[:i]]))
                    start_j = int(np.sum([len(dist) for dist in distributions[:j]]))
                    if precomputed_distances.ndim == 1:
                        precomputed_distances = scipy.spatial.distance.squareform(precomputed_distances)
                    M = precomputed_distances[start_i:start_i+len(distribution_i),start_j:start_j+len(distribution_j)]
                elif w is not None:
                    M = pairwise_mahalanobis_distance_npy(distribution_i,distribution_j,w)

                D[i,j] = ot.emd2([],[],M,numThreads=numThreads)
                #TODO handle non mahalanobis distances
            else:
                D[i,j]=D[j,i]
    
    hardcoded_symbols = None #[i % 4 for i in range(len(distributions))]
    plot_emb(D,method='umap',colors=labels,symbols=hardcoded_symbols,legend=legend,title="UMAP",verbose=True,annotation=None,s=200)
    plot_emb(D,method='diffusion',colors=labels,symbols=hardcoded_symbols,legend=legend,title="DiffMap",verbose=True,annotation=None,s=200)

    hier_clustering(D,labels, ax=None,dist_name="W_θ")
    return D

from sklearn.metrics.pairwise import pairwise_distances

def pairwise_mahalanobis_distance_npy(X_i,X_j=None,w=None,numThreads=32):
    # W has shape dim x dim
    # X_i, X_y have shape n x dim, m x dim
    # return Mahalanobis distance between pairs n x m 
    if X_j is None:
        if w is None or isinstance(w,str):
            return pairwise_distances(X_i,metric=w,n_jobs=numThreads) #cdist .. ,X_j)
        else:
            if w.ndim == 2 and w.shape[0]==w.shape[1]:
                return pairwise_distances(X_i,metric="mahalanobis",n_jobs=numThreads,VI =w)    
            else:
                X_j = X_i
    #Transform poins of X_i,X_j according to W
    #refactor pls
    if w is None or isinstance(w,str):
        return pairwise_distances(X_i,X_j,metric=w,n_jobs=numThreads) #cdist .. ,X_j)
    #elif w.ndim == 2 and w.shape[0]==w.shape[1] and len(w)>1000:
    #mahalanobis matrix is to large to compute distances, use cholesky factorization instead M=wT w
            

            #return pairwise_distances(X_i,metric="mahalanobis",n_jobs=numThreads,VI =w)
            #     
    #return scipy.spatial.distance.squareform(scipy.spatial.distance.cdist(X_i,X_j,metric=w))

    #Assume w is cov matrix of mahalanobis distance
    elif w.ndim == 1:
        #assume cov=0, scale dims by diagonal
        w = np.diag(w)
        proj_X_i = np.matmul(X_i,w)
        proj_X_j = np.matmul(X_j,w)

        #proj_X_i = X_i * w[None,:]
        #proj_X_j = X_j * w[None,:]
    else: 
        w = np.transpose(w)
        proj_X_i = np.matmul(X_i,w)
        proj_X_j = np.matmul(X_j,w)
    
    #print("projected shape")
    #print(proj_X_i.shape)
    return np.linalg.norm(proj_X_i[:,np.newaxis,:]  -  proj_X_j[np.newaxis,:,:],axis=-1)  


# In[3]:


#Synth Data
n = 100
means = [5,10,15]
offsets = np.arange(0,30,3)+1.5

shared_means_x = [0, 40]
shared_means_y = [0, 50] 

use_synth = False
if use_synth:   
    datasets = {
        #"synth_2D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1000,noise_dims=1),
        #"synth_100D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1,noise_dims=99)
        "synth_200D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1,noise_dims=199)
    }

    train_datasets = {
        #"synth_2D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1000,noise_dims=1),
        #"synth_100D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1,noise_dims=99)
        "synth_200D": get_pointcloud(distribution_size=n, class_means = means, offsets = offsets, shared_means_x=shared_means_x, shared_means_y=shared_means_y, plot=True, varying_size=False,return_dict=True,noise_scale=1,noise_dims=199)
    }
else:
    datasets = {}
    train_datasets = {}


# In[4]:


load_scRNA = True
if load_scRNA:
    #Load scRNA datasets
    dataset_folder = "/home/kuehn/ot_metric_learning/damin-ggml/data/datasets/"
    dataset_loading = {
        #"breastcancer":{"path":"b8b5be07-061b-4390-af0a-f9ced877a068","label_col":"reported_diseases","patient_col":"donor_id"},
        #"kidney":{"path":"1c360b0b-eb2f-45a3-aba9-056026b39fa5","label_col":"disease","patient_col":"donor_id"},
        "myocard_infarct":{"path":"c1f6034b-7973-45e1-85e7-16933d0550bc","label_col":"patient_group","patient_col":"sample"}, #major_labls
    }




    
    pca_c = None #50
    max_cells = 1000 #None # 
    subsample_patient_ratio = 1.0

    for dataset_name,loading_info in dataset_loading.items():
        #Test
        data_dict = {}
        data_dict["distributions"],data_dict["distributions_labels"],data_dict["points"], data_dict["point_labels"], data_dict["distribution_modes"], data_dict["patient"] = get_cells_by_patients(dataset_folder+loading_info["path"],loading_info["patient_col"],loading_info["label_col"],subsample_patient_ratio=subsample_patient_ratio,n_feats=pca_c,max_cells = max_cells,filter_genes=True)
        datasets[dataset_name] = data_dict

        #Train
        data_dict2 = {}
        data_dict2["distributions"],data_dict2["distributions_labels"],data_dict2["points"], data_dict2["point_labels"], data_dict2["distribution_modes"], data_dict2["patient"] = get_cells_by_patients(dataset_folder+loading_info["path"],loading_info["patient_col"],loading_info["label_col"],subsample_patient_ratio=0.5,n_feats=pca_c,max_cells = max_cells,filter_genes=True)
        train_datasets[dataset_name] = data_dict2


# In[5]:


metric_params = {}

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[6]:


from metric_learn import LMNN, LFDA, MLKR, NCA, ITML_Supervised
'''
metrics = {
    #"GGML": None,
    "Euclidean": lambda _: "euclidean", # lambda dims:np.eye(dims),
    "Manhatten": lambda _ : "cityblock",
    #"Jaccard": lambda _ : "jaccard",
    "Cosine": lambda _ : "cosine",
    #"LMNN": lambda data:LMNN(n_neighbors=30,random_state=42,learn_rate=1e-6,max_iter=100).fit(data["points"],np.unique(data["point_labels"], return_inverse=True)[1]).get_mahalanobis_matrix(),
    "LFDA": lambda data:LFDA(k=3).fit(data["points"],np.unique(data["point_labels"], return_inverse=True)[1]).get_mahalanobis_matrix(),
    #"NCA" : lambda data: NCA(random_state=42).fit(data["points"],np.unique(data["point_labels"], return_inverse=True)[1]).get_mahalanobis_matrix(),
    
    #"ITML": lambda data: ITML_Supervised(random_state=42).fit(data["points"],np.unique(data["point_labels"], return_inverse=True)[1]).get_mahalanobis_matrix(),
    
}
'''
n_threads = 64

norm='fro'
#Hyperparam  tuning
iter=30
neighbor_t = [3] #,5,7
rank_k = [5] #,10,25,50,100,200] 

alphas = [0.1,1,10,100,1000]
lambdas = [0.1,1,10,100,1000]

metrics = {f"theta_{a}_{l}_3_5_iter{iter}_L{norm}": None for a in alphas for l in lambdas}

#Load params for learned metric
metric_params_path = "/home/kuehn/ot_metric_learning/damin-ggml/data/results/learned_parameters"

#If pipeline should only be rerun for some datasets
selected_datasets = datasets #["breastcancer_full"] #


for d in tqdm(selected_datasets):
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
                    metric_params[d][m]=np.load(f"{metric_params_path}/{d}/GGML/{m}.npy") #hardcoded for hyperparameter tuning of synth 200D
            except Exception as e: 
                print(e)
                metric_params[d][m] = None


# In[7]:


from ggml.benchmark import pivoted_chol

rank = 5

#metric_params_copy= metric_params

#low rank cholesky approximation (pivoted)
for d in metric_params:
    for m in metric_params[d]:
        if isinstance(metric_params[d][m],np.ndarray):
            print(m)
            print(metric_params[d][m].shape)
            if len(metric_params[d][m])>100:
                get_diag = lambda: np.diagonal(metric_params[d][m]).copy()
                get_row = lambda i: metric_params[d][m][i,:]

                metric_params[d][m] = pivoted_chol(get_diag, get_row, rank, err_tol = 1e-6)
                print(metric_params[d][m].shape)


# In[8]:


#wrapper for precomputed distance matrix
#only execute once values are accessed
class Computed_Distances():
    def __init__(self, points, theta, n_threads=n_threads):
        self.n_treads = n_threads
        self.points = points
        self.theta = theta

        
        self.data = np.full((len(points),len(points)), np.nan)

        self.ndim = self.data.ndim
        self.sape = self.data.shape
        #self.shape =  ((len(points),len(points)))

    

    def __getitem__(self, slice_):
        #TODO check if some values are already computed and only compute new values

        if np.isnan(self.data[slice_]).any():
            ranges = [np.squeeze(np.arange(len(self.data))[slice_[i]]) for i in range(len(slice_))] #list(range(slice_[i].stop)[slice_[i]]) for i in range(len(slice_))]
            entry_nan_index = ([],[])
            for entry in ranges[0]:
                #print(self.data[entry,:].ndim)
                check = np.isnan(self.data[entry,:])
                if check.ndim == 2 and np.isnan(self.data[entry,:][:,slice_[1]]).any():
                    entry_nan_index[0].append(entry) 
                elif check.ndim == 1 and np.isnan(self.data[entry,:][slice_[1]]).any(): #TODO why not or?
                    entry_nan_index[0].append(entry)   
            for entry in ranges[1]:
                if np.isnan(self.data[slice_[0],entry]).any():
                    entry_nan_index[1].append(entry)   
            #print("checked for nans")
            
            #check for elements with nan entries
            dist = pairwise_mahalanobis_distance_npy(self.points[entry_nan_index[0],:],self.points[entry_nan_index[1],:],w=self.theta, numThreads = n_threads)
            self.data[np.ix_(entry_nan_index[0],entry_nan_index[1])] = dist 
            #self.data[slice_[0],:][:,slice_[1]] =  pairwise_mahalanobis_distance_npy(self.points[slice_[0],:],self.points[slice_[1],:],w=self.theta, numThreads = n_threads) #self.data[slice_[0],:][:,slice_[1]] should be self.data[slice_]
            #print(self.data[np.ix_(entry_nan_index[0],entry_nan_index[1])]) #self.data[entry_nan_index[0],:][:,entry_nan_index[1]])
            #print(self.data[slice_[0],:][:,slice_[1]])

            computed_percentage = len(entry_nan_index[0])/len(ranges[0])*len(entry_nan_index[1])/len(ranges[1])
            #print(f"loaded {1-computed_percentage:,.2f} of {len(entry_nan_index[0])*len(entry_nan_index[0])} distances")
            return self.data[slice_]

        else:
            return self.data[slice_]

#Precompute ground distances with learned metrics ##actually we are just setting the wrapper instead
precomputed_ground_distances = {}
for d in metric_params:
    print(d)




    precomputed_ground_distances[d]={}
    for m in metric_params[d]:
        print(m)
        #print(datasets[d]["points"].shape)
        #print(metric_params[d][m].shape)
        #print(metric_params[d][m])
       
        if metric_params[d][m] is not None:
            if True:
               
                precomputed_ground_distances[d][m] = Computed_Distances(np.asarray(datasets[d]["points"],dtype='f'),theta=metric_params[d][m]) #pairwise_mahalanobis_distance_npy(np.asarray(datasets[d]["points"],dtype='f'),w=metric_params[d][m])
            #except Exception as e: 
            #    print(e)
            #    print("error")
            #    precomputed_ground_distances[d][m] = None
        else: 
            precomputed_ground_distances[d][m] = None


# In[9]:


def datapoint_acc_over_splits(pred,true,index,length):
    prediction = np.zeros((length,2))
    for s_pred,s_true,s_index in zip(pred,true,index):
        prediction[s_index,0] += np.squeeze([s_pred == s_true]) # 1 #correct
        prediction[s_index,1] += np.squeeze([s_pred != s_true]) #+= 1 #false

    accuracy = prediction[:,0] / np.sum(prediction,axis=-1)
    return accuracy


# In[10]:


import numpy.typing as npt
import warnings

#from pyvoi import VI #TODO

def VI(labels1: npt.NDArray[np.int32],labels2: npt.NDArray[np.int32],torch: bool=True,device: str="cpu",return_split_merge: bool=False):
    """
    Calculates the Variation of Information between two clusterings.

    Arguments:
    labels1: flat int32 array of labels for the first clustering
    labels2: flat int32 array of labels for the second clustering
    torch: whether to use torch, default:True
    device: device to use for torch, default:"cpu"
    return_split_merge: whether to return split and merge terms, default:False

    Returns:
    vi: variation of information
    vi_split: split term of variation of information
    vi_merge: merge term of variation of information
    splitters(optional): labels of labels2 which are split by labels1. splitters[i,0] is the contribution of the i-th splitter to the VI and splitters[i,1] is the corresponding label of the splitter
    mergers(optional): labels of labels1 which are merging labels from labels2. mergers[i,0] is the contribution of the i-th merger to the VI and mergers[i,1] is the corresponding label of the merger
    """
    if labels1.ndim > 1 or labels2.ndim > 1:
        warnings.warn(f"Inputs of shape {labels1.shape}, {labels2.shape} are not one-dimensional -- inputs will be flattened.")
        labels1 = labels1.flatten()
        labels2 = labels2.flatten()
        
    if torch:
        return VI_torch(labels1,labels2,device=device,return_split_merge=return_split_merge)
    else:
        return VI_np(labels1,labels2,return_split_merge=return_split_merge)
    
def VI_np(labels1,labels2,return_split_merge=False):
    assert len(labels2)==len(labels1)
    size=len(labels2)

    mutual_labels=(labels1.astype(np.uint64)<<32)+labels2.astype(np.uint64)

    sm_unique,sm_inverse,sm_counts=np.unique(labels2,return_inverse=True,return_counts=True)
    fm_unique,fm_inverse,fm_counts=np.unique(labels1,return_inverse=True,return_counts=True)
    _,mutual_inverse,mutual_counts=np.unique(mutual_labels,return_inverse=True,return_counts=True)

    terms_mutual = -np.log(mutual_counts/size)*mutual_counts/size
    terms_mutual_per_count=terms_mutual[mutual_inverse]/mutual_counts[mutual_inverse]
    terms_sm = -np.log(sm_counts/size)*sm_counts/size
    terms_fm = -np.log(fm_counts/size)*fm_counts/size
    if not return_split_merge:
        terms_mutual_sum=np.sum(terms_mutual_per_count)
        vi_split=terms_mutual_sum-terms_sm.sum()
        vi_merge=terms_mutual_sum-terms_fm.sum()
        vi=vi_split+vi_merge
        return vi,vi_split,vi_merge

    vi_split_each=np.zeros(len(sm_unique))
    np.add.at(vi_split_each,sm_inverse,terms_mutual_per_count)
    vi_split_each-=terms_sm
    vi_merge_each=np.zeros(len(fm_unique))
    np.add.at(vi_merge_each,fm_inverse,terms_mutual_per_count)
    vi_merge_each-=terms_fm

    vi_split=np.sum(vi_split_each)
    vi_merge=np.sum(vi_merge_each)
    vi=vi_split+vi_merge

    i_splitters=np.argsort(vi_split_each)[::-1]
    i_mergers=np.argsort(vi_merge_each)[::-1]

    vi_split_sorted=vi_split_each[i_splitters]
    vi_merge_sorted=vi_merge_each[i_mergers]

    splitters=np.stack([vi_split_sorted,sm_unique[i_splitters]],axis=1)
    mergers=np.stack([vi_merge_sorted,fm_unique[i_mergers]],axis=1)
    return vi,vi_split,vi_merge,splitters,mergers


# In[11]:


from scipy.cluster.hierarchy import dendrogram
import sklearn
#import pyvoi
'''
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
data = "synth_2D"
metric = "GGML"
labels= np.asarray(datasets[data]["point_labels"])
distances = precomputed_ground_distances[data][metric]
l = 0

for t in np.arange(0.05,1,0.05):
#hier_clustering(distances[:,:],labels)
    print(t)
    pred_cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=len(np.unique(labels)) if l==1 else None,distance_threshold= None if l==1 else np.quantile(distances[:,:],t),metric='precomputed',linkage='average').fit_predict(distances[:,:])

    vi,_,_=VI(pred_cluster,labels,torch=False) #pyvoi.

    print(vi)

    mi_score = sklearn.metrics.mutual_info_score(labels,pred_cluster) #
    ari_score = sklearn.metrics.adjusted_rand_score(labels,pred_cluster) 
    
    #dist = np.copy(distances[:,:])
    #np.fill_diagonal(dist, 0)
    #sh_score = sklearn.metrics.silhouette_score(dist, labels, metric="precomputed")
    print(f"MI:{mi_score:.2f}  ARI:{ari_score:.2f}   VI:{vi:.2f}")
#pred_cluster = pred_cluster.fit(distances[:,:])
                            
#plot_dendrogram(pred_cluster, truncate_mode=None) #"level", p=3)

'''


# In[12]:


patient_group_split = False # True
classify = True
cluster = False# True
results = {}
results_path = "/home/kuehn/ot_metric_learning/damin-ggml/data/results/clustering"

#To plot only cells that can be well rpedicted
datapoints_over_split = {}

#TODO make this setable on a dataset level, and not global
true_modes_available = False


# In[13]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#Classification
import sklearn 
import scipy as sp
from ggml.benchmark import knn_from_dists, plot_table, plot_1split
import pickle

train_size=0.6
test_size =0.2
validation_size= 0.2

for d,data in enumerate(precomputed_ground_distances):
    results[data]={}

    for m, (metric_name, distances) in enumerate(precomputed_ground_distances[data].items()):
        print(f"Data {data} Metric {metric_name}")


        if distances is None:
            results[data][metric_name]=None    
        else:
            
            if True:
            #try:
                points = np.asarray(datasets[data]["points"])
                point_labels = np.asarray(datasets[data]["point_labels"])
                if true_modes_available:
                    points = points[np.asarray(datasets[data]["patient"])==1]
                    point_labels = point_labels[np.asarray(datasets[data]["patient"])==1]
                    print(len(point_labels))
                    #print(len(np.where(np.asarray(datasets[data]["patient"])==1)[0]))
                    mode_ind = np.ix_(np.where(np.asarray(datasets[data]["patient"])==1)[0],np.where(np.asarray(datasets[data]["patient"])==1)[0])

                    element_level = (distances[mode_ind], point_labels)
                    print("computed")
                    #print(distances[mode_ind].shape)
                else:
                    element_level = (distances, point_labels)
                distribution_level = (compute_OT(datasets[data]["distributions"],datasets[data]["distributions_labels"],precomputed_distances=distances,numThreads=n_threads),datasets[data]["distributions_labels"])
            #except Exception as e:
            #    print(e)
            #    results[data][metric_name]=None  

            results[data][metric_name]={"global":{},"ground":{}}
            for l, (distances,labels) in enumerate([element_level,distribution_level]):

                if l==0:
                    neighs = 100
                elif l==1:
                    neighs = 5

                #enforce int labels
                labels = np.unique(labels, return_inverse=True)[1]
    
                
                if True: #try:
                    if classify:
                        print("classification")
                        #Needed for only classifying datapoints that come from distinct modes
                        patient_labels = np.squeeze(np.asarray(datasets[data]["patient"])[np.asarray(datasets[data]["patient"])==1]) #last one hardcoded for synth data, no mdoes known for real world data

                        print(patient_labels.shape)
                        print(labels.shape)
                        #print(distances.shape)

                        pred, true, score, _ , test_indices= knn_from_dists(distances,labels,method=metric_name,weights="uniform",test_size=test_size ,train_size=train_size,n_splits=10,n_neighbors=neighs,distribution_labels= patient_labels if (l==0 and patient_group_split) else None) #'distance' ##, train_indices 
                        result = f"{np.average(score):.2f}±{np.std(score):.2f}"
                        results[data][metric_name]["global" if l==0 else "ground"]["KNN"]=result
                        if l==0:
                            datapoints_over_split[f"{data}_{metric_name}"] = datapoint_acc_over_splits(pred,true,index=test_indices,length=len(labels))

                        print(result)
                    else:
                        results[data][metric_name]["global" if l==0 else "ground"]["KNN"]= None
                else: #except Exception as e:
                    print(e)
                    results[data][metric_name]=None

                try:
                    if cluster:
                        pred_cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=len(np.unique(labels)) if (l==1 or true_modes_available) else None,distance_threshold= None if (l==1 or true_modes_available) else np.quantile(distances[:,:],0.5),metric='precomputed',linkage='average').fit_predict(distances[:,:])
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
    with open(f"{results_path}/{'_'.join(datasets.keys())}2.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{results_path}/{'_'.join(datasets.keys())}2.pickle")




# In[14]:


#results[data][metric_name]["global" if l==0 else "ground"]["MI"]
method_col = metrics.keys()

results_ground_cols = {(a,l):[]  for a in alphas for l in lambdas} #: for d in results.keys() for s in next(iter(results[d].values()))["ground"].keys() }
results_global_cols = {(a,l):[]  for a in alphas for l in lambdas}  #{(d,s):[] for d in results.keys() for s in next(iter(results[d].values()))["global"].keys() }




print(results_ground_cols)

for d in results:
    for a in alphas:
        for l in lambdas:
            m = f"theta_{a}_{l}_3_5_iter{iter}_L{norm}"
            
            if m in results[d] and results[d][m] is not None:
                for s in results[d][m]["ground"]:
                    results_ground_cols[(a,l)].append(results[d][m]["ground"][s])
                for s in results[d][m]["global"]:
                    results_global_cols[(a,l)].append(results[d][m]["global"][s]) 
            else:
                #del results_global_cols[(a,l)]
                results_ground_cols[(a,l)].append("--")
                results_global_cols[(a,l)].append("--")
                #del results_ground_cols[(a,l)]

print(results_ground_cols)

#results_ground_cols["method"] = method_col 
#results_global_cols["method"] = method_col 

print(pd.DataFrame(results_ground_cols))

#normal table
for results_cols in [results_ground_cols,results_global_cols]:
#Ground


    table_df = pd.DataFrame(data=results_cols).transpose()

    new_index = lambdas #table_df.index.levels[0]
    new_columns = alphas #table_df.index.levels[1]
    print(lambdas)
    print(alphas)
    table_df  = pd.DataFrame(
        data=[[results_cols[(n,c)][0] for c in new_columns] for n in new_index],
        index=pd.Index(new_index),
        columns=new_columns,
    )

    #table_df = table_df.reset_index(level=1)
    #display(table_df)
    print(table_df.to_latex(index=True,
                    #formatters={"name": str.upper},
                    float_format="{:.1f}".format,
    ))


#formated for colormap
for results_cols in [results_ground_cols,results_global_cols]:
#Ground


    table_df = pd.DataFrame(data=results_cols).transpose()

    new_index = lambdas #table_df.index.levels[0]
    new_columns = alphas #table_df.index.levels[1]
    table_cmap_df  = pd.DataFrame(
        data=[[f"\\zz{{{results_cols[(n,c)][0].split('±')[0]}}}" for c in new_columns] for n in new_index],
        index=pd.Index(new_index),
        columns=new_columns,
    )

    #table_df = table_df.reset_index(level=1)
    #display(table_cmap_df)
    print(table_cmap_df.to_latex(index=False,
                    #formatters={"name": str.upper},
                    float_format="{:.1f}".format,
    ))
    


# In[15]:


from matplotlib import ticker
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set(font_scale=2)
for i,results_cols in enumerate([results_ground_cols,results_global_cols]):
#Ground


    table_df = pd.DataFrame(data=results_cols).transpose()
    ###
    lambdas_ =  []
    alphas_ = []
    values_ = []
    for l in lambdas: 
        for a in alphas:
            lambdas_.append(l)
            alphas_.append(a)
            values_.append(results_cols[(l, a)][0].split('±')[0])
    wide_df = pd.DataFrame({"alpha":alphas_,"lambda":lambdas_,"values":values_}).replace("--",0.00).astype("f")
    ##

    #pad for contour borders
    new_index = lambdas    #table_df.index.levels[0]
    new_columns = alphas #table_df.index.levels[1]
    table_cmap_df  = pd.DataFrame(
        data=[[f"{results_cols[(n, c)][0].split('±')[0]}" if (n, c) in results_cols else 0 for c in new_columns] for n in new_index],
        index=pd.Index(new_index),
        columns=new_columns,
    )
    table_cmap_df = table_cmap_df.replace("--",0.00).astype("f")

    def pad_m(m):
        if isinstance(m,list):
            #pad coordinates o 10 log scale
            return [m[0]/10] + alphas + [m[-1]*10]
        else:
            return np.pad(m, ((1,1),(1,1)), 'edge')

    fig, ax = plt.subplots()
    ax.plot(np.log10(alphas_),np.log10(lambdas_), "+k")
    cntr = plt.contourf(np.log10(pad_m(new_columns)),np.log10(pad_m(new_index)),pad_m(table_cmap_df.to_numpy()), levels=20, cmap="RdBu_r",corner_mask=True)
    plt.xlim(-1.2,3.2)
    plt.ylim(-1.2,3.2)
    ax.set_xticks(np.log10(alphas))
    ax.set_xticklabels(alphas)
    ax.set_yticks(np.log10(lambdas))
    ax.set_yticklabels(lambdas)
    for axis in [ax.xaxis, ax.yaxis]:
        #formatter.set_scientific(False)
        
        axis.set_major_formatter(ticker.FuncFormatter(lambda y, _: r'$10^{{{:g}}}$'.format(y)))


    plt.rcParams['text.usetex'] = True
    ax.set_ylabel(r"$\lambda$",rotation=0)
    ax.set_xlabel(r"$\alpha$")
    ax.tick_params(axis=u'both', which=u'both',reset=True,right=False,top=False)
    fig.colorbar(cntr, ax=ax, label="accuracy")

    dataset_name="myocard_infarct_full"
    plot_path = f"/home/kuehn/ot_metric_learning/damin-ggml/data/plots/{dataset_name}"
    plt.savefig(plot_path+f"/hyperparams_{'ground' if i == 0 else 'global'}.pdf")

    plt.show()
    '''
    print(wide_df)

    plt.figure()

    cmap = sns.color_palette("light:b", as_cmap=True) #"vlag_r"
    sns.heatmap(table_cmap_df,cmap=cmap, annot=True, fmt=".2f",cbar=False)
    plt.figure()
    sns.kdeplot(
    data=wide_df,  x="alpha",y="lambda",log_scale=10,weights="values",levels=5,fill=True
    )
    plt.xlim(10**-1.2,10**3.2)
    plt.ylim(10**-1.2,10**3.2)

    plt.show()
    #table_df = table_df.reset_index(level=1)
    display(table_cmap_df)
    print(table_cmap_df.to_latex(index=True,
                    #formatters={"name": str.upper},
                    float_format="{:.1f}".format,
    ))
    

    '''


# In[16]:


with open("/home/kuehn/ot_metric_learning/damin-ggml/data/plots/myocard_hyper_ground_results.csv", 'wb') as handle:
        pickle.dump(results_ground_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("/home/kuehn/ot_metric_learning/damin-ggml/data/plots/myocard_hyper_global_results.csv", 'wb') as handle:
        pickle.dump(results_global_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[17]:

'''
with open("/home/kuehn/ot_metric_learning/damin-ggml/data/plots/myocard_hyper_ground_results.csv", 'rb') as handle:
        results_ground_cols = pickle.load(handle)
with open("/home/kuehn/ot_metric_learning/damin-ggml/data/plots/myocard_hyper_global_results.csv", 'rb') as handle:
        results_global_cols = pickle.load(handle)


print(results_global_cols)


# In[18]:
'''

'''
#Global
table_df = pd.DataFrame(data=results_global_cols).transpose()
display(table_df)
print(table_df.to_latex(index=True,
                #formatters={"name": str.upper},
                float_format="{:.1f}".format,
))
'''


# In[ ]:





# In[19]:


#print(precomputed_ground_distances[dataset_name]["GGML"])
print( datapoints_over_split.keys())
dataset_name = "myocard_infarct"
print(precomputed_ground_distances)
considered_metrics = precomputed_ground_distances[dataset_name].keys()


#metric_name = "GGML_L2" #GGML" Euclidean" #

for metric_name in considered_metrics:
    if f"{dataset_name}_{metric_name}" in datapoints_over_split:
        
        thresh = 0. #0.9
        accuracy = datapoints_over_split[f"{dataset_name}_{metric_name}"]
        print(accuracy)
        selected_cells_ind = np.where((accuracy>=thresh))[0] #np.arange((len(accuracy))) #
        print(selected_cells_ind)
        data_points = datasets[dataset_name]["points"][selected_cells_ind]
        print(datasets[dataset_name]["distribution_modes"])
        cell_types = np.concatenate(datasets[dataset_name]["distribution_modes"])[selected_cells_ind]
        #cell_types = [mapping[l] for l in cell_types]

        n_most_cells = 8
        cells_n_min = np.sort(np.unique(cell_types, return_counts=True)[1])[-(n_most_cells-1)]

        unique, counts = np.unique(cell_types, return_counts=True)
        total_other = 0
        celltype_accuracy = {}
        for u,c in zip(unique, counts):
            celltype_accuracy[u]=np.average(accuracy[selected_cells_ind][cell_types==u])
            if c < cells_n_min:
                cell_types[cell_types==u] = "other"
                total_other += c    
        print(celltype_accuracy)
        print(total_other)

        labels = np.asarray(datasets[dataset_name]["point_labels"])[selected_cells_ind]

        theta = metric_params[dataset_name][metric_name]
        print("Surivived until here")

        cell_dist = pairwise_mahalanobis_distance_npy(data_points,w = theta)

        emb1 = plot_emb(cell_dist,method='umap',colors=labels,legend="auto",title=f"{metric_name} UMAP",verbose=True,annotation=None,s=15)
        plot_emb(cell_dist,precomputed_emb=emb1,method='umap',colors=cell_types,legend="auto",title=f"{metric_name} UMAP",verbose=True,annotation=None,s=15)
        plt.show()

