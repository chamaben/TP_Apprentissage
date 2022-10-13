# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import matplotlib.pyplot as plt 
import time 
from sklearn import cluster, metrics
from scipy.io import arff

def search_cluster_kmeans(metric, databrut, k_ref) :
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    plt.scatter(f0 , f1,s=8) 
    plt.show()
    
    tps1 = time.time() 
    if metric == "silhouette" :
        coefs = [99.,99.]
    elif metric == "db" :
        coefs = [99.,99.]
    elif metric == "ch" :
        coefs = [0.,0.]
    print ("Appel KMeans pour une valeur fixee de k ") 
    for k in range(2,15):
        #tps1 = time.time() 
        #k=9
        model = cluster.KMeans(n_clusters=k, init='k-means++') 
        model.fit(datanp) 
        #tps2 = time.time() 
        labels = model.labels_ 
        #print(labels)
        iteration = model.n_iter_ 
        #plt.scatter(f0 , f1, c=labels , s=8) 
        if metric == "silhouette" :
            coefs.append(abs(metrics.silhouette_score(datanp, labels)-1))
        elif metric == "db" :
            coefs.append(metrics.davies_bouldin_score(datanp, labels))
        elif metric == "ch" :
            coefs.append(metrics.calinski_harabasz_score(datanp, labels))
        #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(datanp, labels))
        #plt.title("Donnees apres clustering Kmeans") 
        #plt.show() 
        #print("nb clusters =",k," , nb iter =",iteration , " , ... . . . runtime = ", round((tps2 - tps1)*1000,2),"ms")
    print(coefs)
    plt.plot(coefs)
    plt.show()
    
    
    for i in range(len(coefs)) :
        
        if metric == "silhouette" :
            if coefs[i]==min(coefs):
                nb_cluster = i
                print("silhouette Mectric la plus proche de 1 : ",min(coefs), ", atteint pour nb de cluster = ",i, "k_ref = ", k_ref)
        elif metric == "db" :
            if coefs[i]==min(coefs):
                nb_cluster = i
                print("db Minimum de notre mectric : ",min(coefs), ", atteint pour nb de cluster = ",i, "k_ref = ", k_ref)
        elif metric == "ch" :
            if coefs[i]==max(coefs):
                nb_cluster = i
                print("ch Maximum de notre mectric : ",max(coefs), ", atteint pour nb de cluster = ",i, "k_ref = ", k_ref)
    
    #fait varier k en calculant à chaque fois avec les métriques d'évaluation, si on a une bonne valeur alors on a le bon nb de clusters (k)
    # a refaire sur pls jeu de données
    tps2 = time.time() 
    
    return nb_cluster,round((tps2 - tps1)*1000,2)


def main():
    path = '../artificial/'
    # databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))
    # databrut2 = arff.loadarff(open(path+"xclara.arff",'r'))
    # databrut3 = arff.loadarff(open(path+"sizes4.arff",'r'))
    
    #2.2
    # search_cluster_kmeans("silhouette",databrut, 9)
    # search_cluster_kmeans("db",databrut, 9)
    # search_cluster_kmeans("ch",databrut, 9)
    # search_cluster_kmeans("silhouette",databrut2, 3)
    # search_cluster_kmeans("db",databrut2, 3)
    # search_cluster_kmeans("ch",databrut2, 3)
    # search_cluster_kmeans("silhouette",databrut3,4)
    # avec les différentes métriques on retrouve le bon nb de clusters sauf pour la métrique ch sur le premier exemple
    
    ## 2.3
    databrut4 = arff.loadarff(open(path+"compound.arff",'r'))
    databrut5 = arff.loadarff(open(path+"3-spiral.arff",'r'))
    #search_cluster_kmeans("silhouette",databrut4, 3)
    search_cluster_kmeans("silhouette",databrut5, 3)
    #sur ces deux  jeu de données on remarque QU'ON NE TROUVE PAS LE BON NOMBRE DE CLUSTER AVEC LA MÉTHODE KMEAN
    
main()    