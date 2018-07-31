from metrics import *
import numpy as np

class BisectingClusterer(object):
    def __init__(self, data):
        if data is not None:
            self._data = np.array(data)
            self._I, self._J = self._data.shape
        else:
            print("Empty data")
    
    @property
    def centroids(self):
        return self._centroids

    def fit(self):
        self._centroids = self._compute_centroids_()
        bisecting_indices = self._bisect_clusters_(self._centroids)
        return bisecting_indices
    
    def fit_rows(self):
        return self.fit()
    
    def fit_cols(self):
        data = self._data.T
        aux_I, aux_J = data.shape
        min_correlation = 1
        last_index = 0
        for i in range(aux_I - 1):
            correlation = PositiveNegativeCorrelation(data[i],
                                                      data[+1],
                                                      aux_J).H_pos
            if (correlation<min_correlation):
                min_correlation = correlation
                last_index = i
        indices = np.ones(aux_I)
        zeros = np.zeros(last_index)
        indices[0:last_index] = zeros
        return indices
        
            
    def _compute_centroids_(self):
        max_correlation = 0
        centroids = [0,0]
        for i in range(self._I):
            for j in range(i+1, self._I):
                if (i == j):
                    break
                correlation = PositiveNegativeCorrelation(self._data[i],
                                                          self._data[j],
                                                          self._J).H_neg
                if(correlation > max_correlation):
                    max_correlation = correlation
                    centroids[0] = i
                    centroids[1] = j
        return centroids

    def _bisect_clusters_(self, centroids):
        cluster_indices = np.zeros(self._I)
        for i in range(self._I):
            correlation0 = PositiveNegativeCorrelation(
                self._data[centroids[0]], self._data[i],self._J).H_pos
            correlation1 = PositiveNegativeCorrelation(
                self._data[centroids[1]], self._data[i],self._J).H_pos
            if(correlation0 <= correlation1):
                cluster_indices[i] = 1
        return cluster_indices