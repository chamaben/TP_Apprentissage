import scipy.cluster.hierarchy as shc 
import scipy.cluster
from sklearn import metrics, cluster
import kmedoids
from scipy.io import arff
import time 
import numpy as np 
import matplotlib.pyplot as plt 

def search_dist(method,d_start,d_end, databrut, k_ref) :
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    
    # Le code ci-dessous permet d’afficher un dendrogramme (il y a d’autres possibilités ...) avec la méthode d’agglomération de clusters single

    # Donnees dans datanp 
    print ("Dendrogramme ", method," donnees initiales ") 

    linked_mat = shc.linkage(datanp , method) 


    plt.figure( figsize =(12, 12)) 
    shc.dendrogram(linked_mat , orientation='top' , distance_sort='descending' , show_leaf_counts=False) 
    plt.show()
    
    coefs = []
    cluster_dist = []
    dist = []
    
    for d in range(d_start,d_end):
        
        # Le code suivant permet de déterminer un clustering hiérarchique en utilisant soit une limite sur le seuil de distance soit un nombre de clusters.


        # set distance threshold (0 ensures we compute thge full tree)
        tps1= time.time()
        model= cluster.AgglomerativeClustering(distance_threshold= d, linkage=method, n_clusters=None)
        model=model.fit(datanp)
        tps2=time.time()

        labels= model.labels_
        k= model.n_clusters_
        cluster_dist.append(k)
        leaves=model.n_leaves_

        # affichage clustering
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Résultat du clustering")
        plt.show()
        print(d," nb clusters =",k,", nb feuilles = ", leaves, ", runtime= ", round((tps2-tps1)*1000,2), "ms")

        dist.append(d)
        coefs.append(abs(metrics.silhouette_score(datanp, labels)-1)) 
        
        
    print(coefs)
    plt.plot(coefs)
    plt.show()
    
    for i in range(len(coefs)) :
        if coefs[i]==min(coefs):
            print(method, " silhouette Mectric la plus proche de 1 : ",min(coefs), ", atteint pour nb de cluster = ",cluster_dist[i]," distance ",dist[i],"k_ref = ", k_ref)

def search_cluster_agglomeratif(method,databrut, k_ref) :
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    
    # Le code ci-dessous permet d’afficher un dendrogramme (il y a d’autres possibilités ...) avec la méthode d’agglomération de clusters single

    # Donnees dans datanp 
    tps1 = time.time() 
    print ("Dendrogramme ", method," donnees initiales ") 

    linked_mat = shc.linkage(datanp , method) 


    plt.figure( figsize =(12, 12)) 
    shc.dendrogram(linked_mat , orientation='top' , distance_sort='descending' , show_leaf_counts=False) 
    plt.show()
    
    coefs = [99.0,99.0]
    
    for k in range(2,15):
        
    
        #tps1=time.time()
        model= cluster.AgglomerativeClustering(linkage=method, n_clusters=k)
        model= model.fit(datanp)
        #tps2=time.time()

        labels= model.labels_
        kres= model.n_clusters_
        leaves=model.n_leaves_
        
        # affichage clustering
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Résultat du clustering")
        plt.show()
        
        coefs.append(abs(metrics.silhouette_score(datanp, labels)-1)) 
        
        
        
    print(coefs)
    plt.plot(coefs)
    plt.show()
    
    for i in range(len(coefs)) :
        if coefs[i]==min(coefs):
            nb_cluster = i
            print(method, " silhouette Mectric la plus proche de 1 : ",min(coefs), ", atteint pour nb de cluster = ",i,"k_ref = ", k_ref)

    tps2 = time.time() 
    return nb_cluster,round((tps2 - tps1)*1000,2)

def main():
    path = '../artificial/'
    databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))
    databrut2 = arff.loadarff(open(path+"xclara.arff",'r'))
    databrut3 = arff.loadarff(open(path+"sizes4.arff",'r'))
    #dist varie
    # search_dist("average",10,30,databrut, 9)
    # search_dist("average",30,60,databrut2, 3)
    # search_dist("average",4,9,databrut3,4)
    
    #cluster varie
    # search_cluster_agglomeratif("average",databrut, 9)
    # search_cluster_agglomeratif("average",databrut2, 3)
    # search_cluster_agglomeratif("average",databrut3, 4)
    
    #method varie
    #search_dist("single",1,10,databrut2,3) 
    #search_dist("average",30,60,databrut2,3)
    #search_dist("complete",60,90,databrut2,3)
    #search_dist("ward",500,600,databrut2,3)
    
    
    ## 3.3
    databrut4 = arff.loadarff(open(path+"compound.arff",'r'))
    databrut5 = arff.loadarff(open(path+"3-spiral.arff",'r'))
    search_cluster_agglomeratif("average",databrut4, 3)
    search_cluster_agglomeratif("average",databrut5, 3)
    #sur ces deux  jeu de données on remarque QU'ON NE TROUVE PAS LE BON NOMBRE DE CLUSTER AVEC LA MÉTHODE 
        
#main()