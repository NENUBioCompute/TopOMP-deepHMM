from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from beta_pre_processing import Processor

window_length = 1


processor = Processor()
fasta = "../datasets/test.txt"
hhblits_path = "../datasets/test_hmm/"

x_test = processor.data_pre_processing(fasta, hhblits_path, window_length)

x_test.shape = (x_test.shape[0],  x_test.shape[2])

X_reduced = PCA(n_components=2).fit_transform(x_test)

kmeans = KMeans(n_clusters=3).fit(X_reduced)
print(kmeans)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_, cmap=plt.cm.Set1)
plt.show()