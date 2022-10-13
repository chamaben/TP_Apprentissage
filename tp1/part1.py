import numpy as np
import matplotlib.pyplot as plt

from scipy.io import arff

path = '../artificial/'
databrut = arff.loadarff(open(path+"xclara.arff",'r'))
data =  [[x[0],x[1]] for x in databrut[0]]

f0 = [f[0] for f in data]
f1 = [f[1] for f in data]

plt.scatter(f0,f1,s=8)
plt.title("Donnees initiales")
plt.show()