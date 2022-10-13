
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import  NearestNeighbors
import matplotlib.pyplot as plt 
import time 
from sklearn import cluster, metrics
from scipy.io import arff
import hdbscan

path = '../artificial/'
databrut2 = arff.loadarff(open(path+"xclara.arff",'r'))

datanp =  [[x[0],x[1]] for x in databrut2[0]]
    
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]
plt.scatter(f0 , f1,s=8) 
plt.show()

coefs = [99.,99.]

# # Distances k plus proches voisins
# # Donnees dans X
# k=3
# neigh = NearestNeighbors ( n_neighbors =k )
# neigh . fit ( datanp )
# distances , indices = neigh . kneighbors ( datanp )
# # retirer le point " origine "
# newDistances = np . asarray ([np . average ( distances [i][1:]) for i in range (0 ,
# distances . shape [0])])
# trie = np . sort ( newDistances )

# ##"90-95% des observations qui doivent avoir au moins un voisin dans leur ε-voisinage."
# ##"ε de tel sorte que 90% des observations aient une distance au proche voisin inférieure à ε"
# e= trie[int(len(trie)*0.95)]
# print(e)
# plt . title (" Plus proches voisins (3)")
# plt . plot ( trie ) ;
# plt . show ()


# tps1 = time.time() 
# model = DBSCAN(eps=e, min_samples=5)
# model.fit(datanp) 

# tps2 = time.time() 
# labels = model.labels_ 
# print(labels)
# plt.scatter(f0, f1, c=labels, s=8)
# plt.title("Donnees apres clusturing")
# plt.show()



def search_cluster(databrut, k_ref) :
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    plt.scatter(f0 , f1,s=8) 
    plt.show()
    
    coefs = [99.,99.]
    e_value = [0,0]

        
    for n in range(2,10):
        # Distances k plus proches voisins
        # Donnees dans X
        
        neigh = NearestNeighbors ( n_neighbors =n )
        neigh . fit ( datanp )
        distances , indices = neigh . kneighbors ( datanp )
        # retirer le point " origine "
        newDistances = np . asarray ([np . average ( distances [i][1:]) for i in range (0 ,
        distances . shape [0])])
        trie = np . sort ( newDistances )
        
        ##"90-95% des observations qui doivent avoir au moins un voisin dans leur ε-voisinage."
        ##"ε de tel sorte que 90% des observations aient une distance au proche voisin inférieure à ε"
        e= trie[int(len(trie)*0.95)]
        e_value.append(e)
        print(e)
        plt . title (" Plus proches voisins ")
        plt . plot ( trie ) ;
        plt . show ()
        
        
        tps1 = time.time() 
        model = DBSCAN(eps=e, min_samples=5)
        model.fit(datanp) 
        
        tps2 = time.time() 
        labels = model.labels_ 
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Donnees apres clusturing")
        plt.show()
        
        coefs.append(abs(metrics.silhouette_score(datanp, labels)-1)) 
        

    print(coefs)
    plt.plot(coefs)
    plt.show()
    
    for i in range(len(coefs)) :
        if coefs[i]==min(coefs):
            e_final = e_value[i]
            print(" silhouette Mectric la plus proche de 1 : ",min(coefs), ", atteint pour eps = ",e_value[i],"k_ref = ", k_ref)

    print("test difference min samples")
    
    samples_value = [0,0]
    
    coefs = [99.,99.]
    
    for m in range(2,10):
        
        tps1 = time.time() 
        samples_value.append(m)
        model = DBSCAN(eps=e_final, min_samples=m)
        model.fit(datanp) 
        
        tps2 = time.time() 
        labels = model.labels_ 
        
        plt.scatter(f0, f1, c=labels, s=8)
        plt.title("Donnees apres clusturing")
        plt.show()
        
        coefs.append(abs(metrics.silhouette_score(datanp, labels)-1)) 
        

    print(coefs)
    print(samples_value)
    plt.plot(coefs)
    plt.show()
    
    for i in range(len(coefs)) :
        if coefs[i]==min(coefs):
            print(" silhouette Mectric la plus proche de 1 : ",min(coefs), ", atteint pour min_samples = ",samples_value[i],"k_ref = ", k_ref)


def search_cluster_v2(databrut, k_ref) :
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    plt.scatter(f0 , f1,s=8) 
    plt.show()
    
    
    
    min_c = 99.
    min_e = 99.
    min_s = 99.

        
    for n in range(2,10):
        # Distances k plus proches voisins
        # Donnees dans X
        coefs = [99.,99.]
        
        neigh = NearestNeighbors ( n_neighbors =n )
        neigh . fit ( datanp )
        distances , indices = neigh . kneighbors ( datanp )
        # retirer le point " origine "
        newDistances = np . asarray ([np . average ( distances [i][1:]) for i in range (0 ,
        distances . shape [0])])
        trie = np . sort ( newDistances )
        
        ##"90-95% des observations qui doivent avoir au moins un voisin dans leur ε-voisinage."
        ##"ε de tel sorte que 90% des observations aient une distance au proche voisin inférieure à ε"
        e= trie[int(len(trie)*0.95)]

        print(e)
        plt . title (" Plus proches voisins ")
        plt . plot ( trie ) ;
        plt . show ()
        
        samples_value = [0,0]
        
        for m in range(2,10):
        
            tps1 = time.time() 
            samples_value.append(m)
            model = DBSCAN(eps=e, min_samples=m)
            model.fit(datanp) 
            
            tps2 = time.time() 
            labels = model.labels_ 
            
            plt.scatter(f0, f1, c=labels, s=8)
            plt.title("Donnees apres clusturing")
            plt.show()
            
            coefs.append(abs(metrics.silhouette_score(datanp, labels)-1)) 
        

        print(coefs)
        plt.plot(coefs)
        plt.show()
        
        for i in range(len(coefs)) :
            if coefs[i]==min(coefs) and abs(min(coefs)-1)<abs(min_c-1):
                min_c = coefs[i]
                min_e = e
                min_s = samples_value[i]
                print(" silhouette Mectric la plus proche de 1 : ",min(coefs), ", atteint pour eps = ",e,"min_samples = ",samples_value[i],"k_ref = ", k_ref)


    print(min_c,min_e,min_s)
    model = DBSCAN(eps=min_e, min_samples=min_s)
    model.fit(datanp) 
    labels = model.labels_ 
    print(set(labels))
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title("Donnees apres clusturing final")
    plt.show()
    

def cluster_hdbscan (databrut,k_ref):
    
    datanp =  [[x[0],x[1]] for x in databrut[0]]
    
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    plt.scatter(f0 , f1,s=8) 
    plt.show()
    
    tps1 = time.time()
    model = hdbscan.HDBSCAN()
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_ 
            
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title("Donnees apres clusturing")
    plt.show()
    

#search_cluster_v2(databrut2, 3)


#databrut4 = arff.loadarff(open(path+"compound.arff",'r'))
databrut5 = arff.loadarff(open(path+"3-spiral.arff",'r'))
#search_cluster_v2(databrut4, 3)
search_cluster_v2(databrut5, 3)


## 4.2
# databrut4 = arff.loadarff(open(path+"disk-4600n.arff",'r'))
# databrut5 = arff.loadarff(open(path+"zelnik3.arff",'r'))
# search_cluster_v2(databrut4, 2)
# search_cluster_v2(databrut5, 3)
#sur ces deux  jeu de données on remarqueU'ON NE TROUVE PAS LE BON NOMBRE DE CLUSTER AVEC LA MÉTHODE 
   
# databrut4 = arff.loadarff(open(path+"compound.arff",'r'))
# cluster_hdbscan(databrut4, 3) 
