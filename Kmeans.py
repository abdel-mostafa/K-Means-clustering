import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

x = np.array ([[1, 2],
               [5, 8],
               [1.5, 1.8],
               [8, 8],
               [1, 0.6],
               [9, 11]])
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels) 


for i in range(len(x)):
        print("coordinate:",x[i], "label:", labels[i])
        plt.plot(x[i][0], x[i][1])
