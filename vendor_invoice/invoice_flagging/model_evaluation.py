# model_evaluation.py
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_optimal_k(data):
    sse = []

    for i in range(1, 10):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(data)
        sse.append(km.inertia_)

    plt.plot(range(1, 10), sse)
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title("Elbow Method")
    plt.show()


def train_kmeans(data, k=3):
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(data)
    return model, clusters