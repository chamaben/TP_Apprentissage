from scipy.io import arff
from part2 import search_cluster_kmeans
from part2_4 import search_cluster_kmedoids
from part3_1 import search_cluster_agglomeratif
import os, random



path = '../artificial/'
selected_files = []

##nombre de datasets sur lesquelles on va comparer les méthodes
nb_datasets = 3

temp_moy_kmeans = 0
temp_moy_kmedoids = 0
temp_moy_agglo = 0

cluster_ok =  {"kmeans":0,"kmedoids":0,"agglo":0}



for i in range(nb_datasets):
    
    ##on va comparer les méthodes sur 10 datasets choisis aléatoirement
    random_file=random.choice(os.listdir(path))
    databrut = arff.loadarff(open(path+random_file,'r'))
    
    print(random_file)
    
    ##on récupère le nombre de cluster avec les labels pour vérifier
    if (len(databrut[0][0])>2):
        datanp =  [[x[0],x[1],x[2]] for x in databrut[0]]
        
        f0 = [f[0] for f in datanp]
        f1 = [f[1] for f in datanp]
        f2 = [f[2] for f in datanp]
        f2 = set(f2)
    else :
        continue
    
    print("nb clusters : ", len(f2))
    
    
    selected_files.append(databrut)
    
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
    print("----------------------------------")
    print("file : ",random_file)
    print("real nb of clusters : ",len(f2))
    print("------ nb clusters execution------time")
    print("kmeans : ",nb_cluster_kmeans,"------",t_kmeans)
    print("kmedoids : ",nb_cluster_kmedoids,"------",t_kmedoids)
    print("agglomeratif : ", nb_cluster_agglo,"------",t_agglo)
    print("----------------------------------")
   
##analyses globales : temps d'éxécution moyen, réussite du clustering
print("----------------------------------")
print("Statistics on execution time (average in ms) : ")
print("kmeans : ",temp_moy_kmeans/nb_datasets)
print("kmedoids : ",temp_moy_kmedoids/nb_datasets)
print("agglomeratif : ", temp_moy_agglo/nb_datasets)
print("----------------------------------")
print("Statistics of success : ")
print("kmeans : ",cluster_ok["kmeans"]/nb_datasets)
print("kmedoids : ",cluster_ok["kmedoids"]/nb_datasets)
print("agglomeratif : ", cluster_ok["agglo"]/nb_datasets)
print("----------------------------------")