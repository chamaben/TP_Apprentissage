import matplotlib.pyplot as plt 
import time 
from sklearn import cluster, metrics
from scipy.io import arff

def search_cluster_kmeans(metric, databrut, k_ref) :
    
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    
    #plt.scatter(f0 , f1,s=8) 
    #plt.show()
    
    tps1 = time.time() 
    
    ##On initialise le vecteur de coef sur les deux premières valeurs pour conserver
    ##un index cohérent étant donné que k commence à 2
    
    if metric == "silhouette" :
        coefs = [99.,99.]
    elif metric == "db" :
        coefs = [99.,99.]
    elif metric == "ch" :
        coefs = [0.,0.]
        
    saved_labels =[[],[]]
    
    for k in range(2,15):
        model = cluster.KMeans(n_clusters=k, init='k-means++') 
        model.fit(datanp) 
        labels = model.labels_ 
        saved_labels.append(labels)
        
        # plt.scatter(f0 , f1, c=labels , s=8)
        # plt.title("Donnees apres clustering Kmeans") 
        # plt.show()
        
        ##on calcule les coefs avec la métrique d'évaluation
        if metric == "silhouette" :
            coefs.append(abs(metrics.silhouette_score(datanp, labels)-1))
        elif metric == "db" :
            coefs.append(metrics.davies_bouldin_score(datanp, labels))
        elif metric == "ch" :
            coefs.append(metrics.calinski_harabasz_score(datanp, labels))
          
    # plt.plot(coefs)
    # plt.show()
    
    
    for i in range(len(coefs)) :
        if metric == "silhouette" :
            if coefs[i]==min(coefs):
                nb_cluster = i
                print("Coefficient de silhouette")
                ##on prend la valeur la plus proche de 1
                print("Coef : ",min(coefs), ", atteint pour nb de cluster = ",i, " et k_ref = ", k_ref)
        
        elif metric == "db" :
            if coefs[i]==min(coefs):
                nb_cluster = i
                print("Indice de Davies-Bouldin")
                ##on prend la valeur minimale
                print("Indice : ",min(coefs), ", atteint pour nb de cluster = ",i, " et k_ref = ", k_ref)
        elif metric == "ch" :
            if coefs[i]==max(coefs):
                nb_cluster = i
                print("Indice de Calinski-Harabasz")
                ##on prend la valeur maximale
                print("Indice : ",max(coefs), ", atteint pour nb de cluster = ",i, " et k_ref = ", k_ref)

    tps2 = time.time() 
    
    plt.scatter(f0 , f1, c=saved_labels[nb_cluster] , s=8)
    plt.title("Donnees apres clustering Kmeans") 
    plt.show()
    
    print("temps d'éxécution : ", round((tps2 - tps1)*1000,2))
    
    return nb_cluster,round((tps2 - tps1)*1000,2)


def main():
    path = '../artificial/'
    databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))
    databrut2 = arff.loadarff(open(path+"xclara.arff",'r'))
    databrut3 = arff.loadarff(open(path+"sizes4.arff",'r'))
    
    #2.2
    print("---------------------------")
    print("2d-10c")
    search_cluster_kmeans("silhouette",databrut, 9)
    search_cluster_kmeans("db",databrut, 9)
    search_cluster_kmeans("ch",databrut, 9)
    print("---------------------------")
    print("xclara")
    search_cluster_kmeans("silhouette",databrut2, 3)
    search_cluster_kmeans("db",databrut2, 3)
    search_cluster_kmeans("ch",databrut2, 3)
    print("---------------------------")
    print("sizes4")
    search_cluster_kmeans("silhouette",databrut3,4)
    search_cluster_kmeans("db",databrut3, 4)
    search_cluster_kmeans("ch",databrut3, 4)
    
    # Avec les différentes métriques on retrouve le bon nb de clusters 
    # sauf pour la métrique ch sur le premier exemple et db pour le dernier 
    
    #2.3 : Limites de la méthode
    databrut4 = arff.loadarff(open(path+"compound.arff",'r'))
    databrut5 = arff.loadarff(open(path+"3-spiral.arff",'r'))
    print("---------------------------")
    print("compound")
    search_cluster_kmeans("silhouette",databrut4, 3)
    print("---------------------------")
    print("3-spiral")
    search_cluster_kmeans("silhouette",databrut5, 3)
    
    #Sur ces deux  jeu de données on remarque 
    #que le nombre de clusters trouvés avec kmeans n'est pas bon
    
#main()    