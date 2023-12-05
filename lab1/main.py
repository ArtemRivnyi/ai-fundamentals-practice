import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv('C:\\II labs 3\\Lab01\\lab01.csv', delimiter=';')


# Визначення оптимальної кількості кластерів за допомогою методу зсуву середнього
scores = []
range_values = range(2, 16)

for i in range_values:
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    scores.append(score)

# Побудова графіку
plt.figure(figsize=(10,5))
plt.plot(range_values, scores, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Score')
plt.title('Оцінка кількості кластерів')
plt.show()

# Визначення оптимальної кількості кластерів
optimal_clusters = scores.index(max(scores)) + 2

# Кластеризація методом k-середніх з оптимальною кількістю кластерів
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10)
kmeans.fit(data)

# Виведення результатів
print(f'Оптимальна кількість кластерів: {optimal_clusters}')
print(f'Мітки кластерів: {kmeans.labels_}')

# Вихідні точки на площині
plt.figure(figsize=(10,5))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
plt.title('Вихідні точки на площині')
plt.show()

# Центри кластерів
plt.figure(figsize=(10,5))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('Центри кластерів')
plt.show()

# Кластеризовані дані з областями кластеризації
plt.figure(figsize=(10,5))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('Кластеризовані дані з областями кластеризації')
plt.show()