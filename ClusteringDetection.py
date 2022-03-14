from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as plt


def sel_best( arr, X) -> list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx = np.argsort(arr)[:X]
    return arr[dx]


class ClusterDetection:

    def check_analysis(self, experiences):
        data = dict()
        for experience in experiences:
            current_datapoint = list()
            mote = experience.mote
            if mote not in data.keys():
                data[mote] = list()

            for transmission in experience.state:
                for value in list(transmission.values()):
                    current_datapoint.append(value)

            current_datapoint.append(experience.action)

            for transmission in experience.next_state:
                for value in list(transmission.values()):
                    current_datapoint.append(value)

            data[mote].append()

    def determine_cluster_amount(self,data):
        n_clusters = np.arange(2, 20)
        bics = []
        bics_err = []
        iterations = 20
        for n in n_clusters:
            tmp_bic = []
            for _ in range(iterations):
                gmm = GMM(n, n_init=2).fit(data)

                tmp_bic.append(gmm.bic(data))
            val = np.mean(sel_best(np.array(tmp_bic), int(iterations / 5)))
            err = np.std(tmp_bic)
            bics.append(val)
            bics_err.append(err)
        np.gradient(bics)

        plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
        plt.title("Gradient of BIC Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("grad(BIC)")
        plt.legend()


