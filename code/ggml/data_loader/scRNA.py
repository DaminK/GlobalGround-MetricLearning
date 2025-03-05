## Import the usual libraries
import sys
sys.path.insert(0, '..')
#for local import of parent dict



from tqdm import tqdm
import os
import numpy as np
import scanpy as sc
import anndata as ad

#synth Data
from ggml.generator import get_pointcloud, create_t_triplets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#Optimal Transport
import ot

#Plotting
from ggml.plot import plot_distribution,plot_emb, hier_clustering
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import scipy as sp

class scRNA_Dataset(Dataset):
    #The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, *args, **kwargs):
        #Generate syntehtic data
        distributions, distributions_labels, patient_labels, disease_labels, celltype_node_labels = get_cells_by_patients_2(*args, **kwargs)
        
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


def full_get_cells_by_patients(adata, patient_col="donor_id",label_col="reported_diseases",subsample_patient_ratio=0.25,n_feats=None,filter_genes=False,n_cells=None,**kwargs):
    if isinstance(adata,str):
        adata = sc.read_h5ad(adata)
    

    string_class_labels = np.unique(adata.obs[label_col])

    #detect low variable genes
    if filter_genes:
        gene_var = np.var(adata.X.toarray(),axis=0)

        #filter
        thresh = np.mean(gene_var) #TODO make this not hardcoded and arbitrary
        adata = adata[:,gene_var >thresh]

    #TODO don't let anyone see this 
    distributions = []
    distributions_class = []
    patient_labels = []
    disease_labels = []
    celltype_node_label = []
    dataset_labels = []
    origin_node_label = []
    patient_cell_labels = []



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

        if n_cells is None or patient_adata.n_obs >= n_cells:
            
            if n_cells is not None:
                sc.pp.subsample(patient_adata,n_obs = n_cells) 

            #if n_feats == 50:
            #    p_arr = np.asarray(patient_adata.obsm["X_pca"],dtype="f")
            #else:
            p_arr = np.asarray(patient_adata.X.toarray(),dtype="f") #TODO directly sparse scipy matrix to numpy matrix?


            distributions.append(p_arr)
            
            disease_labels.append(string_class_label)
            distributions_class.append(np.where(string_class_labels==string_class_label)[0][0])
            patient_labels.append(patient)
            patient_cell_labels.append(list(patient_adata.obs['sample']))
            celltype_node_label.append(list(patient_adata.obs['cell_subtype']))
            origin_node_label.append(list(patient_adata.obs['dataset']))

            dataset_labels.append(patient_adata.obs["dataset"][0])
        else:
            print(f"patient {patient} has {patient_adata.n_obs} cells")

    points = np.concatenate(distributions) #np.reshape(np.asarray(dists),(-1,2))
    point_labels = sum([[l] * len(D) for l,D in zip(disease_labels,distributions)],[])
    

    return distributions, distributions_class, points, point_labels,  patient_labels, disease_labels, celltype_node_label, dataset_labels, origin_node_label, patient_cell_labels

def get_cells_by_patients(adata_path,patient_col="donor_id",label_col="reported_diseases",subsample_patient_ratio=0.5,n_feats=None,max_cells = None,filter_genes=False,**kwargs):
    global adata
    adata = sc.read_h5ad(adata_path+".h5ad")  #adata_path+".h5ad")
    print(adata)

    string_class_labels = np.unique(adata.obs[label_col])

    if filter_genes:
        #detect low variable genes
        gene_var = np.var(adata.X.toarray(),axis=0)
        #filter
        thresh = np.mean(gene_var) #TODO make this not hardcoded and arbitrary
        adata = adata[:,gene_var >thresh]
        #adata.write(adata_path+"_filtered.h5ad")

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


#TODO check diffs between cells load funcs of patient
def get_cells_by_patients_2(adata_path,patient_col="donor_id",label_col="reported_diseases",subsample_patient_ratio=0.25,n_cells=1000,pca_components=None,filter_genes=False,**kwargs):
    global adata
    adata = sc.read_h5ad(adata_path)

    string_class_labels = np.unique(adata.obs[label_col])

    #detect low variable genes
    if filter_genes:
        gene_var = np.var(adata.X.toarray(),axis=0)

        #filter
        thresh = np.mean(gene_var) #TODO make this not hardcoded and arbitrary
        adata = adata[:,gene_var >thresh]


    distributions = []
    distributions_class = []
    patient_labels = []
    disease_labels = []
    celltype_node_label = []

    if pca_components is not None:
        pca = PCA(n_components=pca_components, svd_solver='auto')
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
            if pca_components is not None:
                p_arr = pca.transform(p_arr)

            distributions.append(p_arr)
            
            disease_labels.append(string_class_label)
            distributions_class.append(np.where(string_class_labels==string_class_label)[0][0])
            patient_labels.append(patient)
            celltype_node_label.append(list(patient_adata.obs['cell_type']))
            
   

    return distributions, distributions_class, patient_labels, disease_labels, celltype_node_label
