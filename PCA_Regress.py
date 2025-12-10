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
import pickle
from brokenaxes import brokenaxes

def shape_matrix (array):
    """ 
    This function takes in the interpPSTH array and will reshape it from [conditions, neurons, time bins] to a 2D matrix [conditions x timebins, neurons]

    Parameters: 
        array: must be an interpPSTH array which has the shape [conditions, neurons, time bins]

    Returns: 
        new_mat: reshaped the array into a 2 dimension matrix [conditions x timebins, neurons]--making it a tall and skinny array for SVD
    """
    conditions, neurons, time_bins = array.shape
    new_mat = np.zeros((conditions * time_bins, neurons))

    # reshapes the array to insert all time bins for one condition for each neuron, then moves onto the next condition
    for i in range(conditions):
        new_mat[i*time_bins:(i+1)*time_bins, :] = array[i,:,:].T
    return new_mat

def shape_tensor (matrix, conditions, time_bins = None):
    """ 
    This function takes in the interpPSTH matrix [conditions x timebins, neurons] and will reshape it to dimensions [conditions, neurons, time bins]

    Parameters: 
        array: must be an interpPSTH array which has the shape [conditions, neurons, time bins]

    Returns: 
        new_mat: reshaped the array into a 2 dimension matrix [conditions x timebins, neurons]--making it a tall and skinny array for SVD
    """
    time_cond, neurons = matrix.shape

    if time_bins == None: 
        time_bins = int(time_cond/conditions)

    new_tensor = np.zeros((conditions, neurons, time_bins))

    for i in range(conditions):
        low_t = i * time_bins
        high_t = (i + 1) * time_bins
        new_tensor[i,:,:] = matrix[low_t:high_t, :].T
    return new_tensor




def svd (matrix, plot = False): 
    """ 
     This function takes in the interpPSTH 2D matrix [conditions x timebins, neurons] and will compute singular value decomposition (use shape_matrix ()
    before). It will return the left singular vectors, singular values, and right singular vectors and create a plot of the fractional variance by PC if 
    requested.

    Parameters: 
        matrix: must be an interpPSTH array which has the shape [conditions x timebins, neurons]
        plot: must be either True or False value. True will result in a plot formed to show the variance each singular value captures. 

    Returns: 
        U: the left singular vectors 
        S_: the singular values 
        V_T: the right singular vectors 
    """
    # Computing the covariance 
    C_2 = np.cov(matrix.T)

    # Running SVD on the covariance matrix
    U,S_,V_T = np.linalg.svd(C_2)
    s_tot = np.sum(S_)
    frac = S_/s_tot
    

    # Plots the fraction of variance by PC
    if plot: 
        plt.plot([k for k in range(0,len(S_))], frac, linestyle = ' ', marker = 'o')
        plt.xlabel('kth PC')
        plt.ylabel('Fraction of Variance of kth PC')
        plt.ylim(0,1)
        # plt.ylim(0, np.max(S_, 0) + np.max(S_, 0)/10)
        if matrix.shape[1] == 202:
            plt.title('All Neurons Principal Components Fractional Variance')
        elif matrix.shape[1] == 98: 
            plt.title('PMd Principal Components Fractional Variance')
        elif matrix.shape[1] == 104: 
            plt.title('M1 Principal Components Fractional Variance')
        
        plt.show()

    # Returning the left singular vectors, singular values, and right singular vectors 
    return U, S_, V_T

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

    _,S_,_ = svd(matrix)

    
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

def amt_var (matrix, rank): 
    _,S_,_ = svd(matrix)
    
     # initializing variables 
    total_var = np.sum(S_)
    current_var = np.sum(S_[:rank])
    frac = current_var / total_var
    

    print(f'{frac}% variance explained')

def run_PCA (matrix, rank, mc = False):
    """ 
    This function takes in the interpPSTH 2D matrix [conditions x timebins, neurons] and will compute singular value decomposition (use shape_matrix ()
    before). It will then perform a rank k approximation using the specified rank and return the projected data. 

    Parameters: 
        matrix: must be an interpPSTH array which has the shape [conditions x timebins, neurons]
        rank: the specified amount of the dimensions that the data should be projected onto 
        
    Returns:
        proj: the projected rank k approximation of the dataset
        U[:,:rank]: the left singular vectors used to create the approximation 
    """
    
    # runs PCA 
    U, S_, V_T = svd(matrix)

    # create a mean centered matrix
    if mc:
        mean_c = matrix - np.mean(matrix, axis = 0)

        # project the mean centered data onto these PCs to produce a rank k approximation
        proj = mean_c @  U[:, :rank] 
    
    proj = matrix @ U[:,:rank]
    

   
    # takes the dot product of proj_set and V_T to get the rank k approximation, 
    # print(f"proj shape is {proj_set.shape}")
    # print(f"V_T shape is {V_T.shape}")
    proj2 = (U[:, :rank] * S_[:rank]).T @ V_T
    
    return proj, U[:, :rank], proj2

    
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
    _, left_vec, _ = run_PCA(matrix, dimensions)
    
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
    _,left_vec,_ = run_PCA(matrix, dimensions)
    mean_sub = matrix - np.mean(matrix, axis = 0)

    recon = mean_sub @ left_vec @ left_vec.T
    return recon

def mse_fun(true_values, predicted_values):
    """
    This function computes the mean squared error between the true values and the predicted values.

    Parameters:
        true_values: numpy array of true values
        predicted_values: numpy array of predicted values
    """
    return np.mean((true_values - predicted_values) ** 2)

def regress (train_M, train_N, lam):
    """
    This function performs ridge regression on the data matrix M with regularization parameter lam.

    Parameters:
        train_M: numpy array of shape [time-1, rank k] containing rank approximation of the muscle data
        train_N: numpy array of shape [time-1, rank 2k] containing rank approximation of the neural data
        lam: regularization parameter

    Returns:
        W: numpy array of shape [rank 2k, rank k] containing the regression coefficients
        M_hat: numpy array of shape [time-1, rank k] containing the predicted values
        R_squared: numpy array of shape (n_features,) containing the R-squared values for each feature
        MSE: mean squared error of the predictions
    """
    
    # compute the covariance matrix
    C = train_N.T @ train_N
    I = np.eye(C.shape[0])
    

    # compute the weights matrix
    W = np.linalg.solve(C + lam * I, train_N.T @ train_M)
    
    
    # compute the predicted values
    M_hat = train_N @ W
    
    # compute R-squared values
    R_squared = 1 - np.var(train_M - M_hat, axis=0) / np.var(train_M, axis=0)

    return W, M_hat, R_squared

def best_lam(mus_rank, neu_rank, time_bins):
    """
    This function takes in the training data and will compute the best lambda value for ridge regression using cross-validation. It will return the best lambda
    value and the mean squared error for that lambda.

    Parameters: 
        mus_rank: a 2D numpy array of shape [ct, rank] which is the projection onto the first rank PCs
        neu_rank: a 2D numpy array of shape [ct, rank] which is the projection onto the first rank PCs
        M: the original muscle data of shape [conditions x time bins, muscles]

    Returns: 
        best_lambda: the best lambda value found during cross-validation
        mse: the mean squared error for the best lambda
    """

    # conditions in the sample 
    conds = int(neu_rank.shape[0] / time_bins)

    
    # Define a range of lambda values to test
    lambdas = np.logspace(-2, 3, 20)
    
    # Initialize variables to store the best lambda and its corresponding MSE
    best_lambda = None
    min_rmse = float('inf')
    
    #split into testing and training, with a 20/80 split
    test_size = int(np.round(0.2 * conds))
    test_idx = np.random.choice(np.arange(conds), size = test_size, replace = False)
    
    # shaping back into a tensor 
    neu_tensor = shape_tensor(neu_rank, conds)
    mus_tensor = shape_tensor(mus_rank, conds)

    # isolating the test data 
    neu_test_tens = neu_tensor[test_idx, :, :]
    mus_test_tens = mus_tensor[test_idx, :, :]

    # indexes for training data 
    mask = np.ones(conds, dtype = bool)
    mask[test_idx] = False
    neu_training_tens = neu_tensor[mask, :, :]
    mus_training_tens= mus_tensor[mask, :, :]

    # shaping back into matrix for regression
    neu_test = shape_matrix(neu_test_tens)
    mus_test = shape_matrix(mus_test_tens)
    neu_train = shape_matrix(neu_training_tens)
    mus_train = shape_matrix(mus_training_tens)

    mse_vals = []
    rmse_vals = []

    # Perform cross-validation
    for lam in lambdas:
        # Fit a ridge regression model with the current lambda
        W_hat= regress(mus_train, neu_train, lam)[0]

        # Predict on the test sample
        check = neu_test @ W_hat
        
        mse = mse_fun(mus_test, check)
        mse_vals.append(mse)

        rmse = np.sqrt(mse)
        rmse_vals.append(rmse)

        # Update best lambda if current MSE is lower than previous minimu
        
        if rmse < min_rmse and lam != None:
            min_rmse = rmse
            best_lambda = lam
    
    print(">>> best_lam returning:", best_lambda)
    # Return the best lambda and its corresponding MSE         
    return best_lambda, min_rmse

def r_regress (N_tilde, M_tilde, PCs, N_dim = 6, M_dim = 3, num_bins = 236, mc = False): 
    """
    Takes in M and N matrices and runs least squares regression on these matrices projected onto their first N_dim and M_dim PCs
    to generate a weight matrix (W) so that M_hat = N W. Also calculates R squared values. 

    Parameters: 
        M: This is a 2D matrix [conditions x time bins, muscle/neuron readouts] which is dependent on N
        N: This is a 2D matrix [conditions x time bins, neurons] 
        condition: reach number 
        M_dim: amount of PCs to project M onto 
        N_dim: amount of PCs to project N onto (should be double M)
        num_bins: how many time bins are in each trial 

    Returns:
        W: weight matrix found using least squares regression
        M_hat: the product of N_tilde (N projected onto N_dim PCs) and W. Shape is [num_bins, M_dim] 
        M_hat_recon: reconstruction of M using M_hat and the PCs  
        R_squared: one value of R squared for every column of M_hat
    
    """

    # Calling best lambda
    N_tilde_cov = N_tilde.T @ N_tilde
    I = np.identity(N_dim)
    lam, _ = best_lam(M_tilde, N_tilde, num_bins)
    
    # retrieving the weights matrix for M_tilde = W N_tilde and the sum of squares regression
    W = np.linalg.solve(N_tilde_cov + lam * I, N_tilde.T @ M_tilde)

    # calculating the M_hat by multiplying N_tilde and W from above
    M_hat = N_tilde @ W
    
    R_squared = []

    # calculating R squared for each column of M_tilde
    for i in range (W.shape[1]):
        SST = M_tilde[:,i] - np.mean(M_tilde[:,i])
        SST = SST @ SST.T
        SSR = M_hat[:,i] - np.mean(M_tilde[:,i])
        SSR = SSR @ SSR.T
        R_sq = 1 - (SSR / SST)
        R_squared.append(R_sq)
    R_squared = np.array(R_squared)

    # projecting M_hat onto the PCs of M for a reconstruction 
    M_hat_recon = M_hat @ PCs.T 

    # calcualting mean squared error of the reconstruction 
    MSE = mse_fun(M_tilde, M_hat)
    
    
    return W, M_hat, M_hat_recon, R_squared, MSE
    
def scaling (tensor):
    """
    Takes in a tensor of shape [conditions, neurons, time bins] and scales it between 0 and 1. Then returns a tall and skinny 2D matrix of 
    shape [conditions x time bins, neurons]. 

    Parameters: 
        tensor: a 3D tensor of shape [conditions, neurons, time bins]

    Returns: 
        norm_matrix: a 2D version of tensor (shaped [conditions x time bins, neurons]) which is scaled between 0 and 1 
    """

    new_matrix = shape_matrix(tensor)

    # columns max and min 
    col_max = np.amax(new_matrix, axis = 0)
    col_min = np.amin(new_matrix, axis = 0)
  

    # normalizing by their ranges
    norm_matrix = (new_matrix) / (col_max - col_min)

    return(norm_matrix)
    

def fig_3_cut_t(tensor, dimensions):
    """
    This function takes in the interpPSTH 3D tensor [conditions, neurons, time bins] and will create figure 3 from the Churchland et al. 2012 paper. 

    Parameters: 
        tensor: must be an interpPSTH array which has the shape [conditions, neurons, time bins]
        dimensions: the number of dimensions to project onto (should be between 6 and 10)
    """
    # using a tensor with only the preparatory and motor activity
    cut_tensor = time_cut(tensor)
    conditions, _, time_bins = cut_tensor.shape

    # transforming the 3D tensor into a 2D matrix [condition x time, neurons] and scaling and mean centering it 
    matrix = scaling(cut_tensor)
    mean_centered = matrix - np.mean(matrix, axis = 0)

    # gathering the left vectors
    _, left_vec, _ = run_PCA(mean_centered, dimensions)

    # returning the scaled, mean centered, and time cut matrix into a tensor  
    scaled_tensor = shape_tensor(mean_centered, conditions, time_bins)

    # starting the figures
    fig, axs = plt.subplots(dimensions - 1, dimensions -1, figsize=(12, 6))
    axs = axs.flatten()
    c = 0
    
    for i in range(dimensions - 1):
        # dimension 1 for projection
        dim1_vector = left_vec[:,i]
    
        for k in range(dimensions): 

            if k != i: 
                # dimension 2 for projection
                dim2_vector = left_vec[:, k]
               
                for j in range(conditions):
                    current_cond = scaled_tensor[j, :, :]

                    # just making sure it is 2D and not 3D
                    current_cond = current_cond.reshape(scaled_tensor.shape[1], scaled_tensor.shape[2])

                    if i < dimensions - 1 & scaled_tensor.shape[2] == 236:
                        #projecting the data
                        dim1 = current_cond.T @ dim1_vector
                        dim2 = current_cond.T @ dim2_vector

                        # axs[c].plot(dim1[0], dim2[0], 'o', color='gray', markersize=8, label='Start')
                        # axs[c].plot(dim1[1:30], dim2[1:30], '-', color='orange', label='Other')
                        axs[c].plot(dim1[30:81], dim2[30:81], '-', color='blue', label='Preparatory')
                        axs[c].plot(dim1[120], dim2[120], 'o', color='gray', label='Go')
                        axs[c].plot(dim1[150:215], dim2[150:215], '-', color='green', label='Movement')
                        axs[c].plot(dim1[215], dim2[215], 'o', color='red', label='Movement')
                        # axs[c].plot(dim1[215:236], dim2[215:236], '-', color='orange', label='Other')

                        axs[c].set_xlabel(f"Dimension {i + 1}")
                        axs[c].set_ylabel(f"Dimension {k + 1}")

                    elif i < dimensions - 1: 
                        # projecting the data
                        dim1 = current_cond.T @ dim1_vector
                        dim2 = current_cond.T @ dim2_vector

                        axs[c].plot(dim1[:50], dim2[:50], '-', color='blue', label='Preparatory')
                        axs[c].plot(dim1[50], dim2[50], 'o', color='gray', label='Go')
                        axs[c].plot(dim1[51:116], dim2[51:116], '-', color='green', label='Movement')
                        axs[c].plot(dim1[116], dim2[116], 'o', color='red', label='Movement')

                        axs[c].set_xlabel(f"Dimension {i + 1}")
                        axs[c].set_ylabel(f"Dimension {k + 1}")

                c +=1

             

    plt.tight_layout()
    plt.show()


def fig_3_spec(tensor, dimensions, d1, d2):
    """
    This function takes in the interpPSTH 3D tensor [conditions, neurons, time bins] and will create figure 3 from the Churchland et al. 2012 paper. 

    Parameters: 
        tensor: must be an interpPSTH array which has the shape [conditions, neurons, time bins]
        dimensions: the number of dimensions to project onto (should be between 6 and 10)
        d1: the first PC selected for projection (the range is 1 through dimensions)
        d2: the second PC selected for projection (the range is 1 through dimensions)
    """
   # using a tensor with only the preparatory and motor activity
    cut_tensor = time_cut(tensor)
    conditions, _, time_bins = cut_tensor.shape

    # transforming the 3D tensor into a 2D matrix [condition x time, neurons] and scaling and mean centering it 
    matrix = scaling(cut_tensor)
    mean_centered = matrix - np.mean(matrix, axis = 0)

    # gathering the left vectors
    _, left_vec, _ = run_PCA(mean_centered, dimensions)

    # returning the scaled, mean centered, and time cut matrix into a tensor  
    scaled_tensor = shape_tensor(mean_centered, conditions, time_bins)

    # dimension 1 for projection
    dim1_vector = left_vec[:,d1-1]

    # dimension 2 for projection
    dim2_vector = left_vec[:, d2-1]
    
    for j in range(conditions):
        current_cond = scaled_tensor[j, :, :]

        # just making sure it is 2D and not 3D
        current_cond = current_cond.reshape(scaled_tensor.shape[1], scaled_tensor.shape[2])
        
        # projecting the data
        dim1 = current_cond.T @ dim1_vector
        dim2 = current_cond.T @ dim2_vector

        if scaled_tensor.shape[2] == 236:
            plt.plot(dim1[30:81], dim2[30:81], '-', color='blue', label='Preparatory')
            plt.plot(dim1[120], dim2[120], 'o', color='gray', label='Go')
            plt.plot(dim1[150:215], dim2[150:215], '-', color='green', label='Movement')
            plt.plot(dim1[215], dim2[215], 'o', color='red', label='Movement')

            plt.xlabel(f"Dimension {d1}")
            plt.label(f"Dimension {d2}")

        else: 
            plt.plot(dim1[:50], dim2[:50], '-', color='blue', label='Preparatory')
            plt.plot(dim1[50], dim2[50], 'o', color='gray', label='Go')
            plt.plot(dim1[51:116], dim2[51:116], '-', color='green', label='Movement')
            plt.plot(dim1[116], dim2[116], 'o', color='red', label='Movement')

            plt.xlabel(f"Dimension {d1}")
            plt.ylabel(f"Dimension {d2}")

    plt.tight_layout()
    plt.show()



def time_shift(tensor_N, tensor_M, scale = True, mean_c = True, tensors = False):
    """
    This function will both splice the data based on critical time events referenced in the paper. This is 
    necessary before PCA or anything can be run on the data 

    Parameters: 
        tensor_N: This is the inter_PSTH for the N matrix in the equation M = WN
        tensor_M: this is the inter_PSTH for the M matrix in the equation M = WN
        scale: this boolean will scale the data from 0 and 1 if True 
        mean_c : this boolean will mean center the tensor data if True
        tensors: this boolean will return the spliced tensor [conditions, neurons, new time bins], with all other 
            changes selected above   
    
    Return: 
        N_adj_tensor: the inter_PSTH tensor [conditions, neurons, preparatory and movement time] with preparatory activity, 
            movement period activity, and requested scaling and mean centering
        N_move_tensor: the inter_PSTH tensor [conditions, neurons, movement time] with movement period activity, and 
            requested scaling and mean centering
        M_adj_tensor: the inter_PSTH tensor [conditions, neurons/muscle, movement time] with only motor period 
            activity, and requested scaling and mean centering
        N_cut_mc: the inter_PSTH array [conditions x preparatory time & movement time, neurons] with preparatory activity, 
            movement activity, and requested scaling and mean centering
        N_move_mc: the inter_PSTH array [conditions x movement time, neurons] with movement activity, and requested scaling 
            and mean centering
        M_cut_mc: the inter_PSTH array [conditions x movement time, neurons/muscle ] with only movement period 
            activity, and requested scaling and mean centering

    """
    # preparatory index is from -100ms before the targetOn (400ms) and motor activity is looked at -50ms before the 
    # goCue (1550ms) and 600ms after. Motor activity is shifted 50ms later to account for signalling delay and only includes movement 
    # data
    # cutting the N tensor with the times in preparatory period and movement period
    N_prep_start = 30
    N_prep_end = 81 
    N_move_start = 150 
    N_move_end = 216 
    N_idx = np.r_[N_prep_start:N_prep_end, N_move_start:N_move_end]
    N_cut = tensor_N[:,:, N_idx]

    # this is needed for the regression to find W tilde
    N_move = tensor_N[:,:, N_move_start:N_move_end]

    # cutting the M tensor with the times in preparatory period and movement period
    M_move_start = N_move_start + 5
    M_move_end = N_move_end + 5
    M_idx = np.r_[M_move_start:M_move_end]
    M_move = tensor_M[:,:, M_idx]
    
    # shaping it into a matrix in case not scaling 
    N_cut_matrix = shape_matrix(N_cut)
    N_move_matrix = shape_matrix(N_move)
    M_move_matrix = shape_matrix(M_move)
   
    if scale:
        N_cut_scale = scaling(N_cut)
        N_move_scale = scaling(N_move)
        M_move_scale = scaling(M_move)

    if mean_c & scale:
        N_cut_mc = N_cut_scale - np.mean(N_cut_scale, axis = 0)
        N_move_mc = N_move_scale - np.mean(N_move_scale, axis = 0)
        M_move_mc = M_move_scale - np.mean(M_move_scale, axis = 0)
    elif mean_c:
        N_cut_mc = N_cut_matrix - np.mean(N_cut_matrix, axis = 0)
        N_move_mc = N_move_matrix - np.mean(N_move_matrix, axis = 0)
        M_move_mc = M_move_matrix - np.mean(M_move_matrix, axis = 0)
    
    # in case want back in tensor form for mean centered and scaled 
    if tensors:
        N_adj_tensor = shape_tensor(N_cut_mc)
        N_move_tensor = shape_tensor(N_move_mc)
        M_adj_tensor = shape_tensor(M_move_mc)
        return N_adj_tensor, N_move_tensor, M_adj_tensor

    return N_cut_mc, N_move_mc, M_move_mc

def time_cut (tensor, go_cue = True):
    """
    Will splice out the times for preparatory activity and motor activity, used for figure 3. 

    Parameters: 
        tensor: inter_PSTH tensor [conditions, neurons, time bins]
        go_cue: this is a parameter which will include the time point at the go que

    Returns: 
        cut_tensor: inter_PSTH tensor with only time bins during preparatory activity 
    """
    if go_cue:
        N_idx = np.r_[30:80, 120, 150:216]
    else:
        N_idx = np.r_[30:80, 150:216]
    return tensor[:,:, N_idx]

def fig_4 (tensor_N, tensor_M, dimensions = 6):
    """
    
    """
    cond, _, _ = tensor_N.shape

    # scaling, mean centering, and involving only the time periods needed for regression (the movement)
    regress_N, N_move, regress_M = time_shift(tensor_N, tensor_M, tensors = False)
    time_ct = regress_M.shape [0]
    time_ct_neu = regress_N.shape [0]

    # how many time bins are included in the movement period 
    time_bins = int(time_ct / cond)

    # how many time bins are included in the preparatory and movement period 
    time_bins_pm = int(time_ct_neu / cond)

    # difference in bins 
    diff_bin = int((time_bins_pm - time_bins))
    
    # retrieving data projected onto the first N_dim and M_dim PCs
    N_tilde,_,_ = run_PCA(regress_N, dimensions, mc = False)
    M_tilde,PCs,_ = run_PCA(regress_M, int(dimensions/2), mc = False)

    # removing preparatory time bins
    N_tilde_tens = shape_tensor(N_tilde, cond, time_bins_pm)
    N_tilde_tens_reg = N_tilde_tens[:,:,diff_bin:]

    # reshape for ridge
    N_tilde_reg = shape_matrix(N_tilde_tens_reg)

    # running through ridge regression 
    W, M_hat, M_hat_recon, R_squared, MSE = r_regress(N_tilde_reg, M_tilde, PCs, num_bins = time_bins, mc = False)

    return W, M_hat, M_hat_recon, R_squared, MSE, N_tilde


def fig_4_plot (tensor_N, tensor_M, dimensions = 6, basis = 0, potent = True):
    '''
    
    '''
    # calling figure 4 to do the regression
    W, M_hat, M_hat_recon, R_squared, MSE, N_tilde = fig_4(tensor_N, tensor_M, dimensions)
    U, S_val, V = np.linalg.svd(W)

    # potent and null space basis of W 
    W_potent = U[:, :3] 
    W_null = U[:, 3:]

    # low rank neural data projected onto null and potent space of weights 
    N_potent =  N_tilde @ W_potent
    N_null = N_tilde @ W_null
    
    # setting up time for x axis
    prep_time = np.arange(300, 810, 10)
    move_time = np.arange(1500, 2160, 10)
    all_time = np.concatenate((prep_time, move_time))

    # setting up for loop
    cond = tensor_N.shape[0]
    time_bins = int(N_potent.shape[0] / cond)

    if potent:
        fig = plt.figure(figsize=(5, 2))
        bax = brokenaxes(xlims=((300, 800), (1500, 2150)), ylims=((-1.5, 1.5),), hspace=.05) 


        bax.text(500, -1.25, "Test Epoch", ha='center')
        bax.text(1800, -1.25, "Regression Epoch", ha='center')
        bax.set_title(f"Output Potent Dimension {basis + 1}")

        for i in range(cond):
            start = i* time_bins
            end = start + time_bins
            bax.plot(all_time, N_potent[start:end, 0], '-', color='green')
    
    else: 
        fig = plt.figure(figsize=(5, 2))
        bax = brokenaxes(xlims=((300, 800), (1500, 2150)), ylims=((-1.5, 1.5),), hspace=.05) 


        bax.text(500, -1.25, "Test Epoch", ha='center')
        bax.text(1800, -1.25, "Regression Epoch", ha='center')
        bax.set_title(f"Output Null Dimension {basis + 1}")

        for i in range(cond):
            start = i* time_bins
            end = start + time_bins
            bax.plot(all_time, N_null[start:end, 0], '-', color='green')
       



