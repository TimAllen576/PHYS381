# -*- coding: utf-8 -*-
"""
Calculation of Principal Component Analysis


Updated on 21/7/2023

@author: ajm226
"""
import numpy as np
import matplotlib.pyplot as plt


# main code here
def data_gen():
    "Written by ajm226 to generate data to be manipulated"
    x=np.arange(0,6.5,0.5)
    y=x
    
    # create a two dimnesional grid from the vectors (1-D arrays) x and y
    [X,Y]=np.meshgrid(x,y)
    
    # produce three matrices (H1, H2, H3) representing using the 
    H1=100.0*(1.2-0.35*np.sqrt((X-3)*(X-3)+(Y-3)*(Y-3)))
    H2=1022.8-3.6*Y
    H3=1001.2+3.6*X
    
    
    # flatten the spatial patterns into one dimension for later use
    Hflat1=np.reshape(H1,(169,1))
    Hflat2=np.reshape(H2,(169,1))
    Hflat3=np.reshape(H3,(169,1))
    
    
    # define a set of time series 
    W2=np.arange(0.0,1.0,1.0E-03)
    W1=np.sin(10.0*W2)
    W3=np.random.randn(W1.size)*0.1
    
    # form the time-space matrix (2d matrix) which is madeup a combination of the three
    # spatial patterns. This matrix will then be examined using PCA
    X=np.outer(W1,Hflat1)+np.outer(W2,Hflat2)+np.outer(W3,Hflat3)
    
    # add some random noise to show potential benefits of PCA
    
    #np.random.seed(100)  # setting seed for random number generator - this ensures that the smae set of random numbers are used consistently
    Hrand=15.0*np.random.randn(X.shape[0],X.shape[1])  #simulates noise
    X=X+Hrand # adds noise to the artificial dataset
    return X

def Principal_Component_Analysis(data):
    """Finds the principal components of some data
    Normalizes, calculates covariance, uses numpy functions to find eigenvalues and eigenvectors then sorts by size of eigenvalue
    
    Input: data
    Returns: [eigen_values,eigen_vectors,anomaly,covariance] """
    anomaly = np.empty(np.shape(data))
    for (row, col), val in np.ndenumerate(data):
        anomaly[:,col] = data[:,col] - np.mean(data[:,col])
    covariance = np.matmul(np.transpose(anomaly), anomaly)/(len(anomaly)-1)
    unord_eigen_values, unord_eigen_vectors = np.linalg.eigh(covariance)
    sort_mask = np.argsort(unord_eigen_values)
    eigen_values,eigen_vectors = unord_eigen_values[sort_mask], unord_eigen_vectors[:,sort_mask]
    return [eigen_values,eigen_vectors,anomaly,covariance]

def main():
    "Do the thingy"
    y=data_gen()
    [eigen_values,eigen_vectors,anomaly,covariance]=Principal_Component_Analysis(y)
    PC=np.matmul(anomaly,eigen_vectors)
    for j in [-1,-2,-3]:     # plot the last three values in the PCA analysis - note ordering
        plt.figure()
        plt.subplot(2,1,1)
        plt.pcolor(np.reshape(eigen_vectors[:,j],(13,13)))
        plt.subplot(2,1,2)
        plt.plot(PC[:,j])
        plt.savefig('PCA_Test_%01d.png' %j)  # ensures that ordering is nice
    
#main()     # Comment after use to keep imports clean

    