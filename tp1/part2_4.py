from sklearn import metrics, cluster
import kmedoids
from scipy.io import arff
import time 
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.cluster import rand_score



def search_cluster_kmedoids(search_type,metric, databrut, k_ref) :
    
    print("--------------------", search_type)
    
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    # plt.scatter(f0 , f1,s=8) 
    # plt.show()
    
    
    tps1 = time.time() 
    
    ##On initialise le vecteur de coef sur les deux premières valeurs pour conserver
    ##un index cohérent étant donné que k commence à 2
    
    if metric == "silhouette" :
        coefs = [99.,99.]
    elif metric == "db" :
        coefs = [99.,99.]
        
    saved_labels =[[],[]]
    
    for k in range(2,25):
       
        if search_type == "euclidian":
            ##labels avec la distance euclidienne
            distmatrix = euclidean_distances ( datanp )
            fp = kmedoids . fasterpam ( distmatrix , k )
            labels= fp . labels
            
        elif search_type == "manhattan":
            ##labels avec la distance manhattan
            distmatrix_man = manhattan_distances ( datanp )
            fp_man = kmedoids . fasterpam ( distmatrix_man , k )
            labels= fp_man . labels
        
        elif search_type == "kmeans":
            ##labels avec kmeans
            model = cluster.KMeans(n_clusters=k, init='k-means++') 
            model.fit(datanp) 
            labels = model.labels_ 
            
        else :
            labels=[]
            
        saved_labels.append(labels)
        
    
        ##on calcule les coefs avec la métrique d'évaluation
        if metric == "silhouette" :
            coefs.append(abs(metrics.silhouette_score(datanp, labels)-1))
        elif metric == "db" :
            coefs.append(metrics.davies_bouldin_score(datanp, labels))
      
    
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
        
    
    tps2 = time.time() 
    
    plt.scatter(f0 , f1, c=saved_labels[nb_cluster] , s=8)
    
    plt.title("Donnees apres clustering Kmedoids , search type : "+search_type+ " avec k="+str(nb_cluster)) 
    plt.show()
    
    
    print("temps d'éxécution : ", round((tps2 - tps1)*1000,2))
    
    return nb_cluster,round((tps2 - tps1)*1000,2),saved_labels[nb_cluster]


def main ():
    path = '../artificial/'
    databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))
    databrut2 = arff.loadarff(open(path+"xclara.arff",'r'))
    databrut3 = arff.loadarff(open(path+"sizes4.arff",'r'))
    
    # 2.4 
    print("---------------------------")
    print("2d-10c")
    a,b,labels = search_cluster_kmedoids("euclidian","silhouette",databrut, 9)
    a,b,labels_man = search_cluster_kmedoids("manhattan","silhouette",databrut, 9)
    a,b,labels_kmeans = search_cluster_kmedoids("kmeans","silhouette",databrut, 9)
    #a,b,labels = search_cluster_kmedoids("euclidian","db",databrut, 9)
    print("rand_score euclidian vs manhattan : ",rand_score(labels, labels_man))
    print("rand_score euclidian vs kmeans : ",rand_score(labels, labels_kmeans))
    print("mutual info euclidian vs kmeans : ",metrics.mutual_info_score(labels, labels_kmeans))

    print("---------------------------")
    print("xclara")
    a,b,labels = search_cluster_kmedoids("euclidian","silhouette",databrut2, 3)
    a,b,labels_man = search_cluster_kmedoids("manhattan","silhouette",databrut2, 3)
    a,b,labels_kmeans = search_cluster_kmedoids("kmeans","silhouette",databrut2, 3)
    #a,b,labels = search_cluster_kmedoids("euclidian","db",databrut2, 3)
    print("rand_score euclidian vs manhattan : ",rand_score(labels, labels_man))
    print("rand_score euclidian vs kmeans : ",rand_score(labels, labels_kmeans))
    print("mutual info euclidian vs kmeans : ",metrics.mutual_info_score(labels, labels_kmeans))

    print("---------------------------")
    print("sizes4")
    a,b,labels = search_cluster_kmedoids("euclidian","silhouette",databrut3, 4)
    a,b,labels_man = search_cluster_kmedoids("manhattan","silhouette",databrut3, 4)
    a,b,labels_kmeans = search_cluster_kmedoids("kmeans","silhouette",databrut3, 4)
    #a,b,labels = search_cluster_kmedoids("euclidian","db",databrut3, 4)
    print("rand_score euclidian vs manhattan : ",rand_score(labels, labels_man))
    print("rand_score euclidian vs kmeans : ",rand_score(labels, labels_kmeans))
    print("mutual info euclidian vs kmeans : ",metrics.mutual_info_score(labels, labels_kmeans))

    
     
    
 #main()