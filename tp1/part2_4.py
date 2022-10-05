#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:58:39 2022

@author: ingargio
"""

from sklearn import metrics
import kmedoids
from scipy.io import arff
import time 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

path = '../../artificial/'
databrut = arff.loadarff(open(path+"2d-10c.arff",'r'))

datanp =  [[x[0],x[1]] for x in databrut[0]]
 
f0 = [f[0] for f in datanp]
f1 = [f[1] for f in datanp]

tp1= time.time()
k=3
distmatrix= euclidean_distances(datanp)
fp= kmedoids.fasterpam(distmatrix,k)
tps2= time.time()
iter_kmed= fp.n_iter
labels_kmed= fp.labels
print("Loss with FasterPAM:", fp.loss)

plt.scatter(f0, f1, c=labels_kmed, s=8)
plt.title("Donnees apres clusturing Kmedoids")
plt.show()
print("nb clusters=", k, " , nb iter =" , iter_kmed, ", ...... runtime= ", round((tps2 -tp1)*1000,2),"ms" )
