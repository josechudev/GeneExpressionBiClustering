import numpy as np
import math

class PositiveNegativeCorrelation(object):
    def __init__(self, x, y, J):
        self._x = x
        self._y = y
        self._J = J
        self._x_mean = np.mean(x)
        self._y_mean = np.mean(y)
        self._H_pos = None
        self._H_neg = None

    @property
    def H_pos(self):
        if self._H_pos is None:
            # print("Computing H positive...")
            self._H_pos = self._compute_H_pos()
            # print("H positive value: " + str(self._H_pos))
        return self._H_pos

    @property
    def H_neg(self):
        if self._H_neg is None:
            # print("Computing H negative...")
            self._H_neg = self._compute_H_neg()
            # print("H negative value: " + str(self._H_neg))
        return self._H_neg

    def _compute_H_pos(self):
        H_pos = 0
        for j in range(self._J):
            aux = (((self._x[j] - self._x_mean) -
                    (self._y[j] - self._y_mean))/2.0)**2
            H_pos += aux
        H_pos *= 1.0/math.fabs(self._J)
        H_pos = 1 - H_pos
        return H_pos

    def _compute_H_neg(self):
        H_neg = 0
        for j in range(self._J):
            aux = (((self._x[j] - self._x_mean) +
                    (self._y[j] - self._y_mean))/2.0)**2
            H_neg += aux
        H_neg *= 1.0/math.fabs(self._J)
        H_neg = 1 - H_neg
        return H_neg
    
class PairBasedCoherence(object):
    def __init__(self, X):
        self._X = np.array(X)
        self._I, self._J = X.shape
        self._HP = None

    @property
    def HP(self):
        if self._HP is None:
            # print("Calculating Pair based coherence..")
            self._HP = self._compute_HP_()
            # print("Paired based coherence value: " + str(self._HP))
        return self._HP

    def _compute_HP_(self):
        HP = 0
        for i in range(self._I):
            for j in range(i+1, self._I):
                if (i==j): 
                    break
                x = self._X[i]
                y = self._X[j]
                correlation = PositiveNegativeCorrelation(x, y,self._J)
                H0 = correlation.H_pos
                # H0 = max(correlation.H_pos,correlation.H_neg)
                HP += H0
        HP *= math.fabs(2.0)/(math.fabs(self._I)*(math.fabs(self._I)-1)) if self._I > 1 else 0
        
        return HP if HP < 1 else 0
class MSR(object):
    def __init__(self,data):
        self.data=data
        self.n, self.m = data.shape
        self.aiJ = np.mean(data,axis=1)
        self.aIj = np.mean(data,axis=0)
        self.aIJ = np.mean(data)
        self._H = None
        self._HiJ = None
        self._HIj = None
    
    @property
    def H(self):
        if self._H is None:
            
            self._H = self._compute_H()
            
        return 1-self._H
        
    @property
    def HiJ(self):
        if self._HiJ is None:
            self._HiJ = self._compute_HiJ()
        return self._HiJ
    
    @property
    def HIj(self):
        if self._HIj is None:
            self._HIj = self._compute_HIj()
        return self._HIj
    
    def _compute_H(self):
        H = 0
        for i in range(self.n):
            for j in range(self.m):
                H  += (self.data[i,j] - self.aIj[j] - self.aiJ[i] + self.aIJ ) ** 2
        H *= 1.0/(self.n * self.m)       
        return H
    
    def _compute_HiJ(self):
        HiJ = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.m):
                HiJ[i] += ( self.data[i,j] - self.aIj[j] - self.aiJ[i] + self.aIJ )**2
        HiJ *= 1.0/self.m
        return HiJ

    def _compute_HIj(self):
        HIj = np.zeros(self.m)
        for j in range(self.m):
            for i in range(self.n):
                HIj[j] += ( self.data[i,j] - self.aIj[j] - self.aiJ[i] + self.aIJ )**2
        HIj *= 1.0/self.n
        return HIj
    


def three_dimensional_msr(X):
    
    _I = X.shape[0]
    _J = X.shape[1]
    _XiJ = np.mean(X, axis=1)
    _XIj = np.mean(X, axis=0)
    # _XIJ = np.mean(X)
    _XIJ = np.mean(_XiJ, axis = 0)
    acc = 0
    
    for i in range(_I):
        
        tmp = 0
        
        for j in range(_J):
            
            tmp =  (X[i,j] - _XiJ[i] - _XIj[j] + _XIJ)
            tmp = (tmp[0]**2 + tmp[1]**2)
            acc += tmp

    acc = 1-(acc/(_I*_J*2*np.sqrt(2)))
    
    return acc

def three_dimensional_coherence(r1, r2):
    
    aux_list = list()
    aux_list.append(r1)
    aux_list.append(r2)
    
    return three_dimensional_msr(np.array(aux_list))