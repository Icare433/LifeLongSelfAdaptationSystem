from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def sel_best( arr, X) -> list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx = np.argsort(arr)[:X]
    return arr[dx]


class ClusterDetection:

    def reduce_dimensionality(self):
        return None

    def determine_clustering(self, mote_array, data, reducer=PCA(0.99), clusterer = Birch):
        n_clusters = np.arange(2, 15)
        best_sil = -20
        best_model = None
        sils = []
        sils_err = []
        iterations = 20
        data = reducer.fit_transform(data)
        for n in n_clusters:
            tmp_sil = []
            for _ in range(iterations):
                try:
                    labels = clusterer(n_clusters=n).fit_predict(data)
                    sil = metrics.silhouette_score(data, labels, metric='euclidean')
                    tmp_sil.append(sil)
                    if sil > best_sil:
                        best_sil = sil
                        best_model = labels
                except Exception:
                    pass
            val = np.mean(sel_best(np.array(tmp_sil), int(iterations / 5)))
            err = np.std(tmp_sil)
            sils.append(val)
            sils_err.append(err)
        clusters = dict()
        for index in range(len(best_model)):
            if clusters.get(best_model[index]) is None:
                clusters[best_model[index]] = list()
            clusters[best_model[index]].append(mote_array[index])

        return clusters









