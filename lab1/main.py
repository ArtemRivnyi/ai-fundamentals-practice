import pandas as pd
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Загрузите данные из файла ods
data_ods = pd.read_excel('C:\\II labs 3\\Lab01\\lab01.ods', engine='odf')

# Загрузите данные из файла csv
data_csv = pd.read_csv('C:\\II labs 3\\Lab01\\lab01.csv', sep=';')

# Объедините два набора данных
data = pd.concat([data_ods, data_csv])

# Замените пропущенные значения на среднее значение каждого столбца
data = data.fillna(data.mean())

# Определите количество кластеров с помощью метода сдвига среднего
ms = MeanShift()
ms.fit(data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = len(cluster_centers)

# Оцените score для различных вариантов кластеризации
scores = []
for i in range(2, 16):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=0).fit(data)
    score = silhouette_score(data, kmeans.labels_)
    scores.append(score)

# Проведите кластеризацию методом k-средних с оптимальным количеством кластеров
optimal_clusters = scores.index(max(scores)) + 2
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=0).fit(data)

# Выведите следующие графики
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
