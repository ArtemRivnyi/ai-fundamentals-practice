import pandas as pd
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.cluster import estimate_bandwidth
import matplotlib.pyplot as plt

csv_file_path = 'C:\\II labs 3\\Lab01\\lab01.csv'

def load_data(file_path, sep=None):
    data = pd.read_csv(file_path, sep=sep)
    data = data.fillna(data.mean())
    return data

def determine_optimal_clusters(data):
    scores = []
    for i in range(2, 16):
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=0).fit(data)
        score = silhouette_score(data, kmeans.labels_)
        scores.append(score)
    optimal_clusters = scores.index(max(scores)) + 2
    return optimal_clusters, scores

data = load_data(csv_file_path, sep=';')

bandwidth = estimate_bandwidth(data, quantile=0.15, n_samples=500)
ms = MeanShift(bandwidth=bandwidth)
ms.fit(data)
cluster_centers = ms.cluster_centers_

optimal_clusters, scores = determine_optimal_clusters(data)
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=0).fit(data)

plt.figure(figsize=(16, 8))
plt.subplot(221)
plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
plt.title('Вихідні точки на площині')

plt.subplot(222)
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], color='r')
plt.title('Центри кластерів (метод зсуву середнього)')

plt.subplot(223)
plt.bar(range(2, 16), scores)
plt.title('Бар діаграмма score(number of clusters)')

plt.subplot(224)
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('Кластеризовані дані з областями кластеризації')
plt.show()
