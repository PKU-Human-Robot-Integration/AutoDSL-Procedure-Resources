import numpy as np
from tqdm import tqdm
import math

from distribution.gaussian_distribution import N_Gaussian_Distribution as Cluster
from utils.util import normalized_sampling

class DPMM:
    def __init__(self):
        pass

    def cluster(data, feature_dim, iter_times, alpha, regular):
        '''
            Cluster data using a probabilistic model with the expectation-maximization algorithm.

            @Arguments:
                data (list): List of data points.
                feature_dim (int): Dimensionality of the features.
                iter_times (int): Number of iterations.
                alpha (float): Smoothing parameter.
                regular (float): Regularization parameter.

            @Returns:
                result (dict): A dictionary containing the following keys:
                    - "K" (int): Number of clusters.
                    - "label" (list): Cluster labels for each data point.
                    - "log_likelihood_list" (str): Log-likelihood values for each iteration.

            @Functionality:
                - Initializes clusters with one cluster containing all data points.
                - Iteratively updates cluster assignments and parameters using the expectation-maximization algorithm.
                - Computes log-likelihood values for each iteration.
                - Returns the final clustering result.
        '''

        N = len(data)
        if N == 0: 
            return {
                "K": 0,
                "label": [],
                "log_likelihood_list": " ".join(["0" for _ in range(iter_times)])
            }
        data = np.array(data)
        label = np.zeros(len(data))
        cluster = Cluster(feature_dim)
        cluster.set_data_point(data, regular)
        clusters = [cluster]
        log_likelihood_list = []

        # init with all data in one cluster
        for _ in tqdm(range(iter_times), desc="Processing", unit="iteration"):
            # Update data's cluster
            K = len(clusters)

            log_likelihood = 1
            for i in range(N):
                probs = np.zeros(K+1)
                for k in range(K):
                    probs[k] = clusters[k].probability(data[i]) * (clusters[k].N / (N - 1 + alpha))
                probs[K] = alpha / (N - 1 + alpha)
                new_label, normalized_data = normalized_sampling(probs)
                log_likelihood = log_likelihood * normalized_data[int(label[i])]
                label[i] = new_label
            log_likelihood_list.append(math.log(log_likelihood)/N)
            
            # Update cluster's param
            k, clusters = 0, []
            for i in range(K+1):
                data_point = [x for x in range(N) if label[x] == i]
                if len(data_point) > 0:
                    cluster = Cluster(feature_dim)
                    cluster.set_data_point(data[data_point], regular)
                    clusters.append(cluster)
                    label[data_point] = k
                    k += 1
        result = {
            "K": len(clusters),
            "label": label,
            "log_likelihood_list": " ".join([str(num) for num in log_likelihood_list])
        }
        return result