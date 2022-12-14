from scipy.io import arff
from part2 import search_cluster_kmeans
from part2_4 import search_cluster_kmedoids
from part3_1 import search_cluster_agglomeratif
import os, random
import pandas as pd
from part4 import search_cluster_v2, cluster_hdbscan


path = '../dataset-rapport/'

selected_files = []

##nombre de datasets sur lesquelles on va comparer les méthodes
nb_datasets = 7





def compare_others():
    temp_moy_kmeans = 0
    temp_moy_kmedoids = 0
    temp_moy_agglo = 0
    temp_moy_dbscan = 0
    temp_moy_hdbscan = 0

    for i in os.listdir(path):
        
        if i!="y1.txt" :
            databrut = []
            
        
            
            with open(path+i,'r') as f:
                first_list = []
                
                lines = f.readlines()
                
                for line in lines : 
                    first_list.append([float(line.strip().split()[0]),float(line.strip().split()[1])])
                    
                
                databrut.append(first_list)
                databrut.append("txt_files")
            
            print(i)
            
        
            selected_files.append(databrut)
            
            ##on utilise les trois méthodes
            print("--Kmeans--")
            nb_cluster_kmeans, t_kmeans = search_cluster_kmeans("silhouette", databrut, 0)
            print("--Kmedoids--")
            nb_cluster_kmedoids, t_kmedoids, label = search_cluster_kmedoids("euclidian","silhouette", databrut, 0)
            print("--Agglo--")
            nb_cluster_agglo, t_agglo = search_cluster_agglomeratif("average", databrut, 0)
            
            print("--DBSCAN--")
            l, t_dbscan = search_cluster_v2(databrut, 0)
            print("--HDBSCAN--")
            l2, t_hdbscan = cluster_hdbscan(databrut, 0)
            
            print(t_kmeans)
            
            temp_moy_kmeans += t_kmeans
            temp_moy_kmedoids += t_kmedoids
            temp_moy_agglo += t_agglo
            temp_moy_dbscan += t_dbscan
            temp_moy_hdbscan += t_hdbscan
            
            
            ##analyses pour ce fichier : nb de clusters, temps d'éxécution, réussite du clustering
            print("----------------------------------")
            print("file : ",i)
            print("------ nb clusters------execution time")
            print("kmeans : ",nb_cluster_kmeans,"------",t_kmeans)
            print("kmedoids : ",nb_cluster_kmedoids,"------",t_kmedoids)
            print("agglomeratif : ", nb_cluster_agglo,"------",t_agglo)
            print("dbscan : ",t_dbscan)
            print("hdbscan : ",t_hdbscan)
            print("----------------------------------")
       
    ##analyses globales : temps d'éxécution moyen, réussite du clustering
    print("----------------------------------")
    print("Statistics on execution time (average in ms) : ")
    print("kmeans : ",temp_moy_kmeans/nb_datasets)
    print("kmedoids : ",temp_moy_kmedoids/nb_datasets)
    print("agglomeratif : ", temp_moy_agglo/nb_datasets)
    print("dbscan : ", temp_moy_dbscan/nb_datasets)
    print("hdbscan : ", temp_moy_hdbscan/nb_datasets)
    print("----------------------------------")

# on fait le test de y1 à part car il est très lourd
def compare_y1() :
    databrut = []
    with open(path+'y1.txt','r') as f:
        first_list = []
        
        lines = f.readlines()
        
        for line in lines : 
            first_list.append([float(line.strip().split()[0]),float(line.strip().split()[1])])
            
        
        databrut.append(first_list)
        databrut.append("txt_files")
    
 
    ##on utilise les trois méthodes
    print("--Kmeans--")
    nb_cluster_kmeans, t_kmeans = search_cluster_kmeans("silhouette", databrut, 0)
    print("--Kmedoids--")
    nb_cluster_kmedoids, t_kmedoids, label = search_cluster_kmedoids("euclidian","silhouette", databrut, 0)
    print("--Agglo--")
    nb_cluster_agglo, t_agglo = search_cluster_agglomeratif("average", databrut, 0)
    
    print("--DBSCAN--")
    l, t_dbscan = search_cluster_v2(databrut, 0)
    print("--HDBSCAN--")
    l2, t_hdbscan = cluster_hdbscan(databrut, 0)
    
    print(t_kmeans)
    
    
    ##analyses pour ce fichier : nb de clusters, temps d'éxécution, réussite du clustering
    print("----------------------------------")
    print("file : y1.txt")
    print("------ nb clusters------execution time")
    print("kmeans : ",nb_cluster_kmeans,"------",t_kmeans)
    print("kmedoids : ",nb_cluster_kmedoids,"------",t_kmedoids)
    print("agglomeratif : ", nb_cluster_agglo,"------",t_agglo)
    print("dbscan : ",t_dbscan)
    print("hdbscan : ",t_hdbscan)
    print("----------------------------------")

#compare_y1()
compare_others()