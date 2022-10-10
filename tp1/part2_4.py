#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:58:39 2022

@author: ingargio
"""

from sklearn import metrics, cluster
import kmedoids
from scipy.io import arff
import time 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.cluster import rand_score



def search_cluster(metric, databrut, k_ref) :
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    plt.scatter(f0 , f1,s=8) 
    plt.show()
    
    
    
    if metric == "silhouette" :
        coefs = [99.,99.]
        coefs_man = [99.,99.]
        coefs_kmeans = [99.,99.]
    elif metric == "db" :
        coefs = [99.,99.]
        
    for k in range(2,15):
        tps1 = time . time ()
        distmatrix = euclidean_distances ( datanp )
        fp = kmedoids . fasterpam ( distmatrix , k )
        tps2 = time . time ()
        iter_kmed = fp . n_iter
        labels= fp . labels
        
        distmatrix_man = manhattan_distances ( datanp )
        fp_man = kmedoids . fasterpam ( distmatrix_man , k )
        iter_kmed_man = fp_man . n_iter
        labels_man= fp_man . labels
        
        model = cluster.KMeans(n_clusters=k, init='k-means++') 
        model.fit(datanp) 
        labels_kmeans = model.labels_ 
       
        if metric == "silhouette" :
            coefs.append(abs(metrics.silhouette_score(datanp, labels)-1))
            coefs_man.append(abs(metrics.silhouette_score(datanp, labels)-1))
            coefs_kmeans.append(abs(metrics.silhouette_score(datanp, labels_kmeans)-1))
        elif metric == "db" :
            coefs.append(metrics.davies_bouldin_score(datanp, labels))
            
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Donnees apres clusturing Kmedoids")
        plt.show()
        print("nb clusters=", k, " , nb iter =" , iter_kmed, ", ...... runtime= ", round((tps2 -tps1)*1000,2),"ms" )

        
    print(coefs)
    
    if metric=="silhouette":
        print("rand_score euclidian vs manhattan ",rand_score(labels, labels_man))
        print("rand_score ",rand_score(labels, labels_kmeans))
        print("mutual info ",metrics.mutual_info_score(labels, labels_kmeans))
    plt.plot(coefs)
    plt.show()
    
    for i in range(len(coefs)) :
        
        if metric == "silhouette" :
            if coefs[i]==min(coefs):
                print("silhouette Mectric la plus proche de 1 : ",min(coefs), ", atteint pour nb de cluster = ",i, "k_ref = ", k_ref)
          
        elif metric == "db" :
            if coefs[i]==min(coefs):
                print("db Minimum de notre mectric : ",min(coefs), ", atteint pour nb de cluster = ",i, "k_ref = ", k_ref)
   
    #fait varier k en calculant à chaque fois avec les métriques d'évaluation, si on a une bonne valeur alors on a le bon nb de clusters (k)
    # a refaire sur pls jeu de données

path = '../artificial/'
databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))
databrut2 = arff.loadarff(open(path+"xclara.arff",'r'))
databrut3 = arff.loadarff(open(path+"sizes4.arff",'r'))

search_cluster("silhouette",databrut, 9)
search_cluster("db",databrut, 9)
search_cluster("silhouette",databrut2, 3)
search_cluster("db",databrut2, 3)
search_cluster("silhouette",databrut3,4)
search_cluster("db",databrut3,4)

# 2.4 : silhouette sur kmedoids ne fait pas les bons calculs 