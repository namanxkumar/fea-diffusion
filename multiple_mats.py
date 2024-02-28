#or any mesh outputted by our pipeline (save_meshes: True in datagen for examples), you need to make a function that takes in the mesh, the number of regions to divide it into, and divides it into exactly that many random area regions

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS, SpectralClustering
import hdbscan
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

mesh = pv.read('data/qualitativesample/1/1/domain.1.vtk')
print(mesh)
cords = np.array(mesh.points)
print(cords)

# clustering = hdbscan.HDBSCAN(min_cluster_size=2)
# cluster_labels = clustering.fit_predict(cords)
# plt.scatter(cords[:, 0], cords[:, 1], c=cluster_labels, cmap='viridis')
# plt.title(f'HDBSCAN Clustering')
# plt.show()


n = 10
clustering = KMeans(n_clusters=n)
cluster_labels = clustering.fit_predict(cords)
print('clustering centres', clustering.cluster_centers_)
print('clustering centres', clustering.cluster_centers_.reshape(-1,1))
clustering2_centres = KMeans(n_clusters=4)
cluster2_labels_centres = clustering2_centres.fit_predict(clustering.cluster_centers_.reshape(-2, 2)) 



new_labels = np.empty_like(cluster_labels)
for i in range(n):
    # Find the points in this cluster
    points_in_cluster = cluster_labels == i

    # Assign the new label to these points
    new_labels[points_in_cluster] = cluster2_labels_centres[i]

# Now you can plot the original points with the new labels
plt.scatter(cords[:, 0], cords[:, 1], c=new_labels, cmap='viridis')
plt.title(f'KMeans Clustering')
plt.show()