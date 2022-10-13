from scipy.io import arff
from part2 import search_cluster_kmeans
from part2_4 import search_cluster_kmedoids
from part3_1 import search_cluster_agglomeratif
import os, random



path = '../artificial/'
selected_files = []

nb_datasets = 3

temp_moy_kmeans = 0
temp_moy_kmedoids = 0
temp_moy_agglo = 0

cluster_ok =  {"kmeans":0,"kmedoids":0,"agglo":0}



for i in range(nb_datasets):
    
    random_file=random.choice(os.listdir(path))
    databrut = arff.loadarff(open(path+random_file,'r'))
    
    print(random_file)
    
    if (len(databrut[0][0])>2):
        datanp =  [[x[0],x[1],x[2]] for x in databrut[0]]
        
        f0 = [f[0] for f in datanp]
        f1 = [f[1] for f in datanp]
        f2 = [f[2] for f in datanp]
        f2 = set(f2)
    else :
        datanp =  [[x[0],x[1]] for x in databrut[0]]
        
        f0 = [f[0] for f in datanp]
        f1 = [f[1] for f in datanp]
        f2 = 0
    
    print(f2)
    print("nb clusters : ", len(f2))
    
    selected_files.append(databrut)
    nb_cluster_kmeans, t_kmeans = search_cluster_kmeans("silhouette", databrut, len(f2))
    nb_cluster_kmedoids, t_kmedoids = search_cluster_kmedoids("silhouette", databrut, len(f2))
    nb_cluster_agglo, t_agglo = search_cluster_agglomeratif("average", databrut, len(f2))
    
    print(t_kmeans)
    temp_moy_kmeans += t_kmeans
    temp_moy_kmedoids += t_kmedoids
    temp_moy_agglo += t_agglo
    
    if len(f2)==nb_cluster_kmeans :
        cluster_ok["kmeans"]+= 1 
        
    if len(f2)==nb_cluster_kmedoids :
        cluster_ok["kmedoids"]+= 1 
        
    if len(f2)==nb_cluster_agglo :
        cluster_ok["agglo"]+= 1 
    
    print("----------------------------------")
    print("file : ",random_file)
    print("real nb of clusters : ",len(f2))
    print("kmeans : ",nb_cluster_kmeans,t_kmeans)
    print("kmedoids : ",nb_cluster_kmedoids,t_kmedoids)
    print("agglomeratif : ", nb_cluster_agglo,t_agglo)
    print("----------------------------------")
    
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