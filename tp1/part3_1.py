import scipy.cluster.hierarchy as shc 
from sklearn import metrics, cluster
from scipy.io import arff
import time 
import matplotlib.pyplot as plt 

def search_dist(method,d_start,d_end, databrut, k_ref) :
    
    print("----------",method)
    
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
    saved_labels =[]
    
    tps1= time.time()
    
    ##On fait varier la distance d autour de l'endroit qui correspond au bon nombre 
    ##de cluster sur le dendrogramme (analyse graphique)
    for d in range(d_start,d_end):
        
        # Le code suivant permet de déterminer un clustering hiérarchique en utilisant soit une limite sur le seuil de distance soit un nombre de clusters.
        # set distance threshold (0 ensures we compute thge full tree)
        model= cluster.AgglomerativeClustering(distance_threshold= d, linkage=method, n_clusters=None)
        model=model.fit(datanp)

        labels= model.labels_
        saved_labels.append(labels)
        k= model.n_clusters_
        cluster_dist.append(k)
        
        dist.append(d)
        coefs.append(abs(metrics.silhouette_score(datanp, labels)-1)) 
        
        
    
    for i in range(len(coefs)) :
        if coefs[i]==min(coefs):
            nb_cluster = i
            print("Coefficient de silhouette")
            ##on prend la valeur la plus proche de 1
            print("Coef : ",min(coefs), ", clusters : ",cluster_dist[i]," atteint pour une distance = ",dist[i], " et k_ref = ", k_ref)

    tps2=time.time()
    plt.scatter(f0 , f1, c=saved_labels[nb_cluster] , s=8)
    
    plt.title("Donnees apres clustering agglomeratif distance , method : "+method) 
    plt.show()
    
    
    print("temps d'éxécution : ", round((tps2 - tps1)*1000,2))
    
def search_cluster_agglomeratif(method,databrut, k_ref) :
    
    print("-------------",method)
    
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
    
    tps1 = time.time() 
    coefs = [99.0,99.0]
    saved_labels =[[],[]]
    
    ##On fait varier le nombre de cluster k 
    for k in range(2,25):
        
        
        
        tps1_k = time.time()
        
        model= cluster.AgglomerativeClustering(linkage=method, n_clusters=k)
        model= model.fit(datanp)

        labels= model.labels_
        saved_labels.append(labels)
        
        coefs.append(abs(metrics.silhouette_score(datanp, labels)-1)) 
        
        tps2_k = time.time()
        print("temps d'éxécution apres clusturing Agglomeratif pour k=", str(k) , ": ", round((tps2_k - tps1_k)*1000,2))
              
    
    for i in range(len(coefs)) :
        if coefs[i]==min(coefs):
            nb_cluster = i
            print("Coefficient de silhouette")
            ##on prend la valeur la plus proche de 1
            print("Coef : ",min(coefs), ", atteint pour nb de cluster = ",i, " et k_ref = ", k_ref)

    tps2=time.time()
    plt.scatter(f0 , f1, c=saved_labels[nb_cluster] , s=8)
    
    plt.title("Donnees apres clustering agglomeratif distance , method : "+method+" avec k="+str(nb_cluster)) 
    plt.show()
    
    
    print("temps d'éxécution : ", round((tps2 - tps1)*1000,2))
    
    return nb_cluster,round((tps2 - tps1)*1000,2)

def main():
    path = '../artificial/'
    databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))
    databrut2 = arff.loadarff(open(path+"xclara.arff",'r'))
    databrut3 = arff.loadarff(open(path+"sizes4.arff",'r'))
    
    # #distance varie
    print("Distance varie")
    print("---------------------------")
    print("2d-10c")
    search_dist("average",10,30,databrut, 9)
    print("---------------------------")
    print("xclara")
    search_dist("average",30,60,databrut2, 3)
    print("---------------------------")
    print("sizes4")
    search_dist("average",4,9,databrut3,4)
    
    # #nombre de cluster varie
    print("Nombre de cluster varie")
    print("---------------------------")
    print("2d-10c")
    search_cluster_agglomeratif("average",databrut, 9)
    print("---------------------------")
    print("xclara")
    search_cluster_agglomeratif("average",databrut2, 3)
    print("---------------------------")
    print("sizes4")
    search_cluster_agglomeratif("average",databrut3, 4)
    
    print("---------------------------")
    print("Method varie sur xclara")
    ##method varie sur xclara, on adapte les distance à chaque fois en fonction de la méthode
    search_dist("single",8,12,databrut2,3) ##CA MARCHE PAS
    search_dist("average",30,60,databrut2,3)
    search_dist("complete",60,90,databrut2,3)
    search_dist("ward",500,600,databrut2,3)
    
    
    ## 3.3 : Limites
    databrut4 = arff.loadarff(open(path+"compound.arff",'r'))
    databrut5 = arff.loadarff(open(path+"3-spiral.arff",'r'))
    print("---------------------------")
    print("compound")
    search_cluster_agglomeratif("average",databrut4, 3)
    print("---------------------------")
    print("3-spiral")
    search_cluster_agglomeratif("average",databrut5, 3)
    
main()