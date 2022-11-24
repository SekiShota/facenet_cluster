from PCA import PCA_2d
from sklearn.cluster import KMeans

K=3
reduced=PCA_2d()

kmeans=KMeans(n_clusters=K, max_iter=300)
kmeans.fit(reduced)
pred=kmeans.predict(reduced)

print("result:",pred)
