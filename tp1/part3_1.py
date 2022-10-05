import scipy.cluster.hierarchy as shc 
import scipy.cluster
from sklearn import metrics
import kmedoids
from scipy.io import arff
import time 
import numpy as np 
import matplotlib.pyplot as plt 

path = '../../artificial/'
databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))

datanp =  [[x[0],x[1]] for x in databrut[0]]
 
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]


# Le code ci-dessous permet d’afficher un dendrogramme (il y a d’autres possibilités ...) avec la méthode d’agglomération de clusters single

# Donnees dans datanp 
print ("Dendrogramme 'single' donnees initiales ") 

linked_mat = shc.linkage(datanp , 'single') 


plt.figure( figsize =(12, 12)) 
shc.dendrogram(linked_mat , orientation='top' , distance_sort='descending' , show_leaf_counts=False) 
plt.show()


# Le code suivant permet de déterminer un clustering hiérarchique en utilisant soit une limite sur le seuil de distance soit un nombre de clusters.


# set distance threshold (0 ensures we compute thge full tree)
tps1= time.time()
model= scipy.cluster.AgglomerativeClustering(distance_threshold= 10, linkage='single', n_clusters=None)
model=model.fir(datanp)
tps2=time.time()

labels= model.labels_
k= model.c_clusters_
leaves=model.n_leaves_

# affichage clustering
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Résultat du clustering")
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, ", runtime= ", round((tps2-tps1)*1000,2), "ms")

# set the number of clusters 
k=4
tps1=time.time()
model= scipy.cluster.AgglomerativeClustering(linkage='single', n_clusters=k)
model= model.fit(datanp)
tps2=time.time()

labels= model.labels_
k= model.c_clusters_
leaves=model.n_leaves_


