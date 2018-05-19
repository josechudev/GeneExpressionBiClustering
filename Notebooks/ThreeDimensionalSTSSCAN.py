
# coding: utf-8

# # Three-dimensional STSSCAN 

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import glob
import multiprocessing
import copy

from operator import attrgetter
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from metrics import *


class Bicluster(object):
    
    def __init__(
        self, 
        data, 
        rho=None,
        gamma=None, 
        bisected=False,
        H_max=None, 
        best_bisection=None
    ):
        
        """
        Bicluters structure
        
        
        :param rho: <np.array[int]> the map of the indices of rows of the bicluster i.e [2,4,7,8]
        :param gamma: <np.array[int]> the map of gammas, same structure f rho
        :param bisected: <boolean> indicates if the bicluster has been bisected recentky
        :param h_max: <float> best h from the last bisection
        :paran best_bisection: <np.array[int]> mapping array from the best bisection
        
        *Note: Noise will be represented as -1
        
        """
        
        self.gamma = gamma
        self.rho = rho
        self.bisected = bisected
        self.h_max = H_max
        self.bisected_direction = best_bisection.flg_direction if best_bisection is not None else None
        self.best_bisection = best_bisection
        self.data = data[np.ix_(self.rho, self.gamma)]

    def set_bisected(self):
        
        self.bisected = not self.bisected
    
    def __str__(self):
        
        return ("rho: \n"+ str(self.rho) + "\ngamma: \n" + str(self.gamma) + "\nH: \n " + str(self.H))
                
    def row_length(self):
        
        return len(self.rho>=0)
    
    def col_length(self):
        
        return len(self.gamma>=0)
    
    def set_fields(self, best_bisection, h_max, bisected_direction):
        
        """
        Method
        
        :param best_bisection: <np.array[int]> Mapping array from the best bisection 
        :param h_max: <float> Best H from the last bisection
        :bisected_direction: <String> Direction from the last bisection ('cols' or 'rows')
        """
        
        self.best_bisection = best_bisection
        self.h_max = h_max
        self.bisected_direction = bisected_direction
        final_map = best_bisection == 0 # filtering noise then zeros

        if(bisected_direction == 'cols'):
            
            self.gamma = self.gamma[final_map]
            print(self.gamma)
            
        else:
            
            self.rho = self.rho[final_map]
            
    def compute_H(self, data):
        
        """
        Method:
        Computes the coherence H of the data
        """
        
        self.data = data[np.ix_(self.rho, self.gamma)]
        self.H = three_dimensional_msr(self.data)
        
        return self.H


# In[44]:


def split_cols(data, min_coherence):
    
    """
    Function:
    Spits the bicluster according to the best split in of columns
    
    :param data: <np.array[np.array]> data of the bicluster to split
    :param bicluster: <Bicluster> data 
    :param min_coherence: <float> minimal coherence of a bicluster
    """
    
    t_data = data.transpose(1,0,2)
    aux_I, aux_J = t_data.shape[0:2]
    aux_indices = np.zeros(aux_I)
    middle = 0
    max_h = 0
    temp_max = 0
    flg_noise_cleaning = False
    temp_h_right = 0
    temp_h_left = 0
    
    for i in range(1, aux_I-1):

        temp_h_right = three_dimensional_msr(t_data[i:len(t_data)])
        temp_h_left = three_dimensional_msr(t_data[0:i+1])
        temp_max = max(temp_h_right, temp_h_left)
        
        if (temp_max >= max_h):
            
            max_h = temp_max
            middle = i
            
    indices = np.ones(aux_I)
    zeros = np.zeros(middle)
    
    if len(indices) == 1 or temp_h_left <= min_coherence:
        indices = np.array(-1*np.ones(len(indices)))
        flg_noise_cleaning = True
        
    if len(zeros) == 1 or temp_h_right <= min_coherence:
        zeros = np.array(-1*np.ones(len(zeros)))
        flg_noise_cleaning = True
        
    indices[0:middle] = zeros
    indices = np.array(indices)  
    unique_indices = np.unique(indices)
    
    return (indices, max_h, flg_noise_cleaning)


# In[45]:


def transform_dm(dm):
    dm[dm < 0] = 0
    dm = 1-dm**4
    np.fill_diagonal(dm, 0)
    
    return dm


# In[46]:


def split_rows(data, min_cluster_size):
    
    """
    Apply DBSCAN over the bicluster to filter noise
    
    :param bicluster: <Bicluster> data
    :param min_cluster_size
    """
    
    n_cols = data.shape[1]
    dm = np.asarray([[((three_dimensional_coherence(p1, p2))                        if (three_dimensional_coherence(p1, p2))                        != 1                        else 0.0) 
                      for p2 in data]\
                     for p1 in data])
    dm = transform_dm(dm)
    gmm = GaussianMixture(n_components=3)
    gmm.fit(np.array([(dm).flatten()]).T)
    eps = gmm.means_[np.where(gmm.weights_                              ==(min(gmm.weights_)))].flatten()
    dev = gmm.covariances_.flatten()[np.where(gmm.weights_                                              ==(min(gmm.weights_)))]
    lower_bound = 0 #eps - 8*dev
    rows = np.array([])
    
    
    while (eps) > lower_bound:
        
        db = DBSCAN(eps = eps ,metric='precomputed', 
                    min_samples=min_cluster_size,
               ).fit(dm)
        
        # print(db.labels_)
        rows = db.labels_
        
        if(len(np.unique(db.labels_))<3):
            
            eps -= dev/4
            if eps <= 0:
                
                eps = 1.0e-15
                break
        else:
            
            break

    map_array = rows
    tmp_bicluster_label = 0
    max_h = 0
    
    for i in np.unique(rows):
        
        if i == -1:
            
            pass
        
        else:
            
            temp_h = three_dimensional_msr(data[np.where(rows == i)])
            
            if temp_h >= max_h:
                max_h = temp_h
                tmp_bicluster_label = i

    final_array = np.array([1 if value == tmp_bicluster_label else (0 if value != -1 else -1) 
                            for value in db.labels_])
    map_array = final_array
    flg_noise_cleaning = False
    unique_array = np.unique(final_array)
    
    if (len(unique_array) == 2 and unique_array[0] == -1):
        
        flg_noise_cleaning = True
        
    return (map_array, max_h, flg_noise_cleaning)


# In[47]:


def compute_bicluster_coherence(bicluster, 
                                min_cluster_row_size,
                                min_coherence):
    
    """
    Splits bicluster and chooses wether its better to split rows or columns
    
    :param bicluster: <Bicluster> Bicluster to split
    :param min_coherence: <float> minimal coherence of a bicluster
    """
    
    print("Compute Coherence")
    
    flg_noise_cleaning_rows = False
    flg_noise_cleaning_cols = False
    
    if (bicluster.bisected):
    
        return (bicluster.best_bisection, bicluster.h_max, bicluster.bisected_direction)
    else:
        
        map_array_cols, h_cols, flg_noise_cleaning_cols = split_cols(bicluster.data, min_coherence=min_coherence)
        map_array_rows, h_rows, flg_noise_cleaning_rows = split_rows(bicluster.data, min_cluster_row_size)
        # print("h cols " + str(h_cols) + " h rows " + str(h_rows))
        
        if (h_cols > h_rows):
            
            if flg_noise_cleaning_cols:
                
                return (map_array_cols, h_cols, "clean cols")
            else:
 
                return (map_array_cols, h_cols, "cols")
        else:
            
            if flg_noise_cleaning_rows:
                
                return (map_array_rows, h_rows, "clean rows")
            
            return (map_array_rows, h_rows, "rows")


# In[48]:


class STSSCAN(object):

    def __init__(self, data, n_clusters=2, min_coherence = 0.9):
        """
        STSSCAN
        
        Applies biclustering over data
        
        :param data: <np.array> The data to apply biclusters
        :param n_clusters: <int> The number of biclusters to find
        
        """

        assert data is not None, 'Empty data'
        self._data = data
        (self._I, self._J) = self._data.shape[0:2]
        self._n_clusters = n_clusters
        self._biclusters = list()
        self._objective_function = 0
        self._min_coherence = min_coherence

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def data(self):
        return self._data

    @property
    def biclusters(self):
        return self._biclusters

    def fit(self):
        """
        Fits the data on the algorithm. Iters over the 
        number of biclusters until find the requested number
        """

        print ('Fitting data ')
        n_iterations = 0
        n_clusters = self._n_clusters
        initial_bicluster = Bicluster(self._data, np.arange(self._I),
                np.arange(self._J))
        self._biclusters.append(initial_bicluster)
        
        while len(self._biclusters) != self.n_clusters:
            
            print('\nN iteration ' + str(n_iterations))
            
            self._split_biclusters() 
            n_iterations += 1
        # self._prune()

    def _split_biclusters(self):
        """
        Splits bicluters in order to find the next biclusters that optimizes
        the sum of coherences among biclusters
        """

        # import ipdb; ipdb.set_trace()

        print()
        print('Splitting biclusters ')

        h_max = 0
        tmp_bicluster_delta = 0
        tmp_bisection = None
        tmp_split = 'cols'
        tmp_bisection_direction = None
        position = 0
        best_position = 0
        best_bisection_array = np.array([])
        best_bisection_direction = None

        while position != len(self._biclusters):

            bicluster = self._biclusters[position]
            
            print()
            
            print('Position ' + str(position))
            
            print('\nRow length ' + str(bicluster.row_length()) + ' Col length ' + str(bicluster.col_length()) + '\n')
            
            min_cluster_row_size = math.floor(np.log(self._I) * 5)

            if bicluster.row_length() >= min_cluster_row_size and bicluster.col_length() > 2:

                (tmp_bisection, tmp_bicluster_delta, tmp_split) = compute_bicluster_coherence(bicluster, min_cluster_row_size, self._min_coherence)

                if tmp_split == 'clean cols':

                    index_map = tmp_bisection.copy() >= 0
                    rho = self._biclusters[position].rho.copy()
                    new_gamma = self._biclusters[position].gamma.copy()[index_map]
                    self._biclusters[position] = Bicluster(self._data,
                            rho, new_gamma)
                    best_bisection_direction = 'clean'
                    
                    print('\nCleaned cols at ' + str(position))
                    
                elif tmp_split == 'clean rows':

                    index_map = tmp_bisection.copy() >= 0
                    rho = self._biclusters[position].rho.copy()[index_map]
                    new_gamma = self._biclusters[position].gamma.copy()
                    self._biclusters[position] = Bicluster(self._data,
                            rho, new_gamma)
                    best_bisection_direction = 'clean'
                    print('\nCleaned rows at ' + str(position))
                else:

                    if tmp_bicluster_delta >= h_max:
                        
                        h_max = tmp_bicluster_delta
                        best_position = position
                        best_bisection_array = tmp_bisection.copy()
                        best_bisection_direction = ('cols'
                                 if tmp_split == 'cols' else 'rows')
            position += 1

        if best_bisection_direction != 'clean':
            
            print('\nBisected bicluster at ' + str(best_position))
            best_bisection = best_bisection_array
            self._add_bicluster(best_position, best_bisection, h_max,
                                best_bisection_direction)

    def _add_bicluster(
        self,
        best_position,
        best_bisection,
        h_max,
        bisection_direction,
        ):
        
        """
        Adds the found bicluster into the list of labels
        
        :param best_position: <int> position of the best bicluster
        :param best_bisection: <np.array[int]> map_array of the best bisection
        :param max_delta: h_max 
        """

        # Create the new bicluster

        best_rho = self._biclusters[best_position].rho
        best_gamma = self._biclusters[best_position].gamma
        index_map = best_bisection >= 1
        aux_index_map = best_bisection == 0
        
        if bisection_direction == 'cols':
            
            new_rho = best_rho.copy()
            new_gamma = best_gamma[index_map].copy()
            aux_rho = best_rho.copy()
            aux_gamma = best_gamma[aux_index_map].copy()
            
            # print('\nAdded bicluster cols: ' + str(new_gamma))
            
        else:
            
            new_gamma = best_gamma.copy()
            new_rho = best_rho[index_map].copy()
            aux_rho = best_rho[aux_index_map].copy()
            aux_gamma = best_gamma.copy()
            
            # print('\nAdded bicluster rows: ' + str(new_rho))
            
        new_bicluster = Bicluster(self._data, new_rho, new_gamma)

        self._biclusters.append(new_bicluster)

        # Modify the bicluster that is going to be bisected

        self._biclusters[best_position] = Bicluster(self._data, aux_rho, aux_gamma)

    def _prune(self):

        for position in range(0, len(self._biclusters)):

            print(position)

            rho = self._biclusters[position].rho
            gamma = self._biclusters[position].gamma

            if len(gamma) > 0:

                stds = np.std(self._data[np.ix_(rho, gamma)], axis=0)
                std = np.mean(stds)
                lst_map = list()

                print('std ' + str(std))

                for index in [0, len(stds) - 1]:
                    
                    if stds[index] <= std * 2:
                        
                        lst_map.append(1)
                    else:
                        
                        lst_map.append(0)

                print(lst_map)
    
    def merge(self):
        
        index = 0
        
        for index in range(0,len(self._biclusters)):
            
            sub_index = 0
            
            for sub_index in range(index+1,len(self._biclusters)):
                
                if (max(self._biclusters[index].gamma) == min(self._biclusters[sub_index].gamma) and 
                    self._biclusters[index].rho == self._biclusters[sub_index].rho
                   ):
                    
                    print("Merge " + str(index) + " with " + str(sub_index)) 



