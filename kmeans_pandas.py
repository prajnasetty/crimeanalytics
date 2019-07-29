from sklearn.cluster import KMeans
from sklearn import decomposition
from numpy import genfromtxt
import numpy as numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = genfromtxt('20_Victims_of_rape.csv', delimiter=',')
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
#print(kmeans.cluster_centers_)

