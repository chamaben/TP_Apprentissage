from scipy.io import arff
from part2 import search_cluster_kmeans
from part2_4 import search_cluster_kmedoids
from part3_1 import search_cluster_agglomeratif
import os, random

def compare(databrut,temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok) :
    
    #on récupère le nombre de cluster avec les labels pour vérifier
    if (len(databrut[0][0])>2):
        datanp =  [[x[0],x[1],x[2]] for x in databrut[0]]
        
        f0 = [f[0] for f in datanp]
        f1 = [f[1] for f in datanp]
        f2 = [f[2] for f in datanp]
        f2 = set(f2)

    
    print("nb clusters : ", len(f2))
    
    ##on utilise les trois méthodes
    nb_cluster_kmeans, t_kmeans = search_cluster_kmeans("silhouette", databrut, len(f2))
    nb_cluster_kmedoids, t_kmedoids, label = search_cluster_kmedoids("euclidian","silhouette", databrut, len(f2))
    nb_cluster_agglo, t_agglo = search_cluster_agglomeratif("average", databrut, len(f2))
    
    print(t_kmeans)
    
    temp_moy_kmeans += t_kmeans
    temp_moy_kmedoids += t_kmedoids
    temp_moy_agglo += t_agglo
    
    ##on rajoute 1 si on a bien calculé le nombre de clusters
    if len(f2)==nb_cluster_kmeans :
        cluster_ok["kmeans"]+= 1 
        
    if len(f2)==nb_cluster_kmedoids :
        cluster_ok["kmedoids"]+= 1 
        
    if len(f2)==nb_cluster_agglo :
        cluster_ok["agglo"]+= 1 
    
    ##analyses pour ce fichier : nb de clusters, temps d'éxécution, réussite du clustering

    print("real nb of clusters : ",len(f2))
    print("------ nb clusters------execution time")
    print("kmeans : ",nb_cluster_kmeans,"------",t_kmeans)
    print("kmedoids : ",nb_cluster_kmedoids,"------",t_kmedoids)
    print("agglomeratif : ", nb_cluster_agglo,"------",t_agglo)
    print("----------------------------------")
    
    return temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo,cluster_ok

path = '../artificial/'

print("Datasets qui fonctionnent")
print("-----------------------------------")

temp_moy_kmeans = 0
temp_moy_kmedoids = 0
temp_moy_agglo = 0

cluster_ok =  {"kmeans":0,"kmedoids":0,"agglo":0}

print("2d-10c")
databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))
temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok = compare(databrut,temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok)

print("xclara")
databrut2 = arff.loadarff(open(path+"xclara.arff",'r'))
temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok = compare(databrut2,temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok)

print("sizes4")
databrut3 = arff.loadarff(open(path+"sizes4.arff",'r'))
temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok = compare(databrut3,temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok)

##analyses globales : temps d'éxécution moyen, réussite du clustering
print("----------------------------------")
print("Statistics on execution time (average in ms) : ")
print("kmeans : ",temp_moy_kmeans/3)
print("kmedoids : ",temp_moy_kmedoids/3)
print("agglomeratif : ", temp_moy_agglo/3)
print("----------------------------------")
print("Statistics of success : ")
print("kmeans : ",cluster_ok["kmeans"]/3)
print("kmedoids : ",cluster_ok["kmedoids"]/3)
print("agglomeratif : ", cluster_ok["agglo"]/3)
print("----------------------------------")

print("Datasets qui ne fonctionnent pas")
print("-----------------------------------")

temp_moy_kmeans_pb = 0
temp_moy_kmedoids_pb = 0
temp_moy_agglo_pb = 0
cluster_ok =  {"kmeans":0,"kmedoids":0,"agglo":0}

print("compound")
databrut4 = arff.loadarff(open(path+"compound.arff",'r'))
temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok = compare(databrut4,temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok)

print("3-spiral")
databrut5 = arff.loadarff(open(path+"3-spiral.arff",'r'))
temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok = compare(databrut5,temp_moy_kmeans,temp_moy_kmedoids,temp_moy_agglo, cluster_ok)
    
print("----------------------------------")
print("Statistics on execution time (average in ms) : ")
print("kmeans : ",temp_moy_kmeans/3)
print("kmedoids : ",temp_moy_kmedoids/3)
print("agglomeratif : ", temp_moy_agglo/3)
print("----------------------------------")
print("Statistics of success : ")
print("kmeans : ",cluster_ok["kmeans"]/3)
print("kmedoids : ",cluster_ok["kmedoids"]/3)
print("agglomeratif : ", cluster_ok["agglo"]/3)
print("----------------------------------")

    
    
