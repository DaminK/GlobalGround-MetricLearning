import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt



def get_pointcloud(distribution_size=100, class_means = [0,5,10], offsets = [0,5,10,15], shared_means_x = [], shared_means_y = [], plot=True, varying_size=True,noise_scale=1000,noise_dims=1,return_dict=False,*args, **kwargs):

    #Gaussian along dim 1, uniform along dim 2 (only information is the mean of the gaussian)
    unique_label = np.arange(len(class_means),dtype=int)

    distributions = []
    distributions_labels = []
    plotting_df =[]

    label_distribution_modes = []

    for mean,label in zip(class_means,unique_label):
        i = 0
        for offset in offsets:
            rand_size= np.random.randint(20,distribution_size) if varying_size else distribution_size

            dim1 = np.random.normal(10+mean,size=rand_size,scale=1.5)
            dim2 = np.random.uniform(7.5+offset,12.5+offset,size=(rand_size,noise_dims))

            label_distribution_modes = label_distribution_modes + [1]*rand_size

            for shared_mean_x,shared_mean_y in zip(shared_means_x,shared_means_y):
                dim1 = np.concatenate((dim1,np.random.normal(shared_mean_x,size=rand_size,scale=1.5)))
                dim2 = np.concatenate((dim2,np.random.normal(shared_mean_y,size=(rand_size,noise_dims),scale=1.5)),axis=0) # #np.random.normal(2.5+offset,size=n)
                label_distribution_modes = label_distribution_modes + [0]*rand_size

            dim1 = dim1 * 5 / 4
            dim2 = dim2*noise_scale

            stacked = np.insert(dim2,0,dim1,axis=1)

            #stacked = np.append(dim2,[dim1],axis=0)
            #stacked = np.stack((dim1,dim2),axis=-1)
            plotting_df.append(pd.DataFrame({'x':dim1,'y':dim2[:,0],'class':label,'sample':i}))

            distributions.append(stacked)
            distributions_labels.append(label)
        
            i+=1


    if plot:
        df = pd.concat(plotting_df, axis=0)

        plt.figure(figsize=(6,5))
        ax = sns.scatterplot(df,x='x',y='y',hue="class",style='sample')
        sns.move_legend(ax, "center right", bbox_to_anchor=(1.3, 0.5))

        #xticks = ax.xaxis.get_major_ticks()
        #xticks[0].label1.set_visible(False)
        #yticks = ax.yaxis.get_major_ticks()
        #xticks[-1].label1.set_visible(False)

        plt.show()


    points = np.concatenate(distributions) #np.reshape(np.asarray(dists),(-1,2))
    point_labels = sum([[l] * len(D) for l,D in zip(distributions_labels,distributions)],[]) #flattens list of lists

    if return_dict:
        data_dict = {}
        data_dict["distributions"],data_dict["distributions_labels"],data_dict["points"], data_dict["point_labels"], data_dict["patient"] = distributions, distributions_labels, points, point_labels, label_distribution_modes
        return data_dict
    else:
        return distributions, distributions_labels, points, point_labels, label_distribution_modes

def create_triplets(distributions,labels):
    triplets = []
    for i,_ in enumerate(distributions):
        for j,_ in enumerate(distributions):
            for k,_ in enumerate(distributions):
                if labels[i]==labels[j] and labels[j] != labels[k] and i != j:
                    triplets.append((i,j,k))
    return triplets

def create_t_triplets(distributions,labels,t=5,**kwargs):
    print(f"passed neighs: {t}")
    labels= np.asarray(labels)
    triplets = []
    replace = any(np.unique(labels,return_counts=True)[1]<t)
    

    def get_neighbors(class_,skip=None):
        #get t elements from distributions where labels = class
        #TODO optional skip self
        return np.random.choice(np.where(labels == class_)[0],size=t,replace= replace)
    
    for j,_ in enumerate(distributions):

        c_j = labels[j]
        for i in get_neighbors(c_j):
            for c_k in np.unique(labels):
                if c_k != c_j:
                    for k in get_neighbors(c_k):
                        triplets.append((i,j,k))
    return triplets

