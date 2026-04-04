import math
import os
import scipy
from scipy.optimize import lsq_linear
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, halfnorm
import random
import h5py
from scipy.io import loadmat
import random
import pickle
import sys
sys.path.append(r"c:\Users\katie\OneDrive\Documents\GitHub\trial")
import PCA_Regress as pcar



def plot_PSTH (matrix, start_time = 0, cond = 1, approximation = False, reconstruction = 0, start_PC = 1, ax = None, regression = 0):
    """
    This function takes in the interpPSTH 2D matrix [timebins, neurons] and will plot the PSTH


    Parameters:
        matrix: must be an interpPSTH array which has the shape [timebins, neurons]
        start_time: the beginning time of the data passed through in ms
        cond: which condition is being plotted
        approximation: whether it's a reduced-rank PSTH
        reconstruction: int, used to label reconstructions
        start_PC: first principal component number for labels
        ax: matplotlib.axes.Axes object (optional)
    """
    if ax is None:
        fig, ax = plt.subplots()
   
    # forms the time bins
    num_times = matrix.shape[0]
    times = np.arange(start_time, num_times * 10 + start_time, 10)


   
     # Plot data
    if approximation:
        ax.set_title(f"Reduced Rank PSTH of M1 Reach {cond}")
        for i in range(matrix.shape[1]):
            ax.plot(times, matrix[:, i], label=f"PC {start_PC}")
            start_PC += 1
        if matrix.shape[1] < 7:
            ax.legend()
    else:
        ax.set_title(f"Original PSTH of Reach {cond}")
        for i in range(matrix.shape[1]):
            ax.plot(times, matrix[:, i])


    if reconstruction > 0:
        ax.set_title(f"{reconstruction} Dim Reconstruction of PSTH Reach {cond}")


    if regression > 0:
        ax.set_title(f" Ridge Regression {regression} Dim Reconstruction Reach {cond}")


    # Vertical lines for cues
    cues = [400, 1200, 1550]
    labels = ['target on', 'go cue', 'movement start']
    colors = ['r', 'g', 'y']
    for xval, label, color in zip(cues, labels, colors):
        ax.axvline(x=xval, color=color, linestyle='--')
        ax.text(xval + 1, ax.get_ylim()[0] * 0.96, label,
                rotation=0, verticalalignment='center',
                color=color, fontweight='bold')


   
    ax.set_xlabel('time (ms)')
    ax.set_xlim(start_time, times[-1])
    ax.set_ylabel('spikes per second')
    if (np.max(matrix) < 1.1):
        ax.set_ylabel('scaled spikes per second')
    ax.grid(True)
   
def projections(matrix, dimensions):
    _, left_vec = pcar.run_PCA(matrix, dimensions)
   
    # matrix_cent = matrix - np.mean(matrix, axis=0)
    dim1 = matrix @ left_vec[:, 0]
    rows = int(np.ceil(dimensions / 2))


    fig, axs = plt.subplots(2, rows, figsize=(12, 6))
    axs = axs.flatten()


    # Set up a single set of labels and line handles for the legend


    legend_lines = []
    legend_labels = ['Start', 'Other', 'Preparatory', 'Movement']


    for i in range(dimensions - 1):
        dim_temp = matrix @ left_vec[:, i + 1]


        axs[i].plot(dim_temp[0], dim1[0], 'o', color='red', markersize=8, label='Start')
        axs[i].plot(dim_temp[1:30], dim1[1:30], '-', color='blue', label='Other')
        axs[i].plot(dim_temp[30:70], dim1[30:70], '-', color='orange', label='Preparatory')
        axs[i].plot(dim_temp[70:135], dim1[70:135], '-', color='blue', label='Other')
        axs[i].plot(dim_temp[135:215], dim1[135:215], '-', color='green', label='Movement')
        axs[i].plot(dim_temp[215:236], dim1[215:236], '-', color='blue', label='Other')


        axs[i].set_xlabel(f"Dimension {i + 2}")
        axs[i].set_ylabel("Dimension 1")


    plt.tight_layout()
    plt.show()
   


def neu_recon (matrix, dimensions):
    _,left_vec = pcar.run_PCA(matrix, dimensions)
    mean_sub = matrix - np.mean(matrix, axis = 0)


    recon = mean_sub @ left_vec @ left_vec.T
    return recon


def amt_var (matrix, rank):
    _,S_,_ = pcar.svd(matrix)
   
     # initializing variables
    total_var = np.sum(S_)
    current_var = np.sum(S_[:rank])
    frac = current_var / total_var
   


    print(f'{frac}% variance explained')


def frac_var (matrix, ideal_var, plot = False):
    """
    This function takes in the interpPSTH 2D matrix [conditions x timebins, neurons] (use shape_matrix ()before). It will then call svd() to compute the
    singular values. Next it will compute how many PCs are needed to acquire the variance the user has input (value between 0 and 1). It will plot
    a graph of the cumulative singular values if requested, and print out how many PCs are needed for the requested variance.  


    Parameters:
        matrix: must be an interpPSTH array which has the shape [conditions x timebins, neurons]
        ideal_var: can be any number between 0 and 1, representing the amount of variation the user would like the PCs to capture from the data
        plot: must be either True or False value. True will result in a plot formed to show the cumulative variance for singular values.


    """
    # the ideal_var should be passed over as a value between 0 and 1


    _,S_,_ = pcar.svd(matrix)


   
    # initializing variables for the for loop
    total_var = np.sum(S_)
    frac = []
    current_var = 0


    # fills frac with the cumulative variance when using the singular values including that index (index i contains the cumulative variance for the
    # :i singular values
    for i in range(len(S_)):
        current_var += S_[i]
        frac.append(current_var/total_var)


    # this will identify how many PCs would be needed to capture the preferred variance
    for k in range (len(S_)):
        if frac[k] >ideal_var:
            print("index for ideal variance is ", k)
            break
   


    # will produce a plot for the cumulative variance
    if plot:
        plt.plot([k for k in range(0,len(S_))], frac, linestyle = ' ', marker = 'o')
        plt.axvline(x=k, color='r', linestyle='-', label=f'{ideal_var}% variance explained')
        plt.xlabel('kth PC')
        plt.ylabel('cum fraction of kth PC')
        plt.ylim(0,1)
        plt.legend()
        if matrix.shape[1] == 202:
            plt.title('All Neurons Principal Components Cumulative Fractional Variance')
        elif matrix.shape[1] == 98:
            plt.title('PMd Principal Components Cumulative Fractional Variance')
        elif matrix.shape[1] == 104:
            plt.title('M1 Principal Components Cumulative Fractional Variance')
       
        plt.show()
