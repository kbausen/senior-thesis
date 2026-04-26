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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
import pickle
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec


def shape_matrix (tensor):
    """
    This function takes in the interpPSTH array and will reshape it from [conditions, neurons, time bins] to a 2D matrix [conditions x timebins, neurons]


    Parameters:
        array: must be an interpPSTH tensor which has the shape [conditions, neurons, time bins]


    Returns:
        new_mat: reshaped the array into a 2 dimension matrix [conditions x timebins, neurons]--making it a tall and skinny array for SVD
    """
    conditions, neurons, time_bins = tensor.shape
    new_mat = np.zeros((conditions * time_bins, neurons))

    # reshapes the tensor to insert all time bins for one condition for each neuron, then moves onto the next condition
    for i in range(conditions):
        new_mat[i*time_bins:(i+1)*time_bins, :] = tensor[i,:,:].T
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


def ident (tensor_N):
    """
    Identifies what dataset is being used for corresponding time cuts and titles


    Parameters:
        tensor_N: This is the inter_PSTH tensor [conditions, neurons, time] for the N matrix in the equation M = WN


    Returns:
        J: boolean which specifies if this is dataset J or not
        PMd: boolean which specifies if this is a PMd tensor or includes all neurons
    """

    # identifying dataset N or J to ensure correct time splits and titles
    cond, neu, fin_tim = tensor_N.shape
    if fin_tim < 229:
        J = False
    else:
        J = True
   
    if J:
        if neu > 100:
            PMd = True
        else:
            PMd = False
    else:
        if neu < 150:
            PMd = True
        else:
            PMd = False
   
    return J, PMd

def J_split (J_PMd_tensor, J_M1_tensor): 
    J_PMd_1 = J_PMd_tensor[:27, :, :]
    J_PMd_2 = J_PMd_tensor[27:54, :, :]
    J_PMd_3 = J_PMd_tensor[54:81, :, :]
    J_PMd_4 = J_PMd_tensor[81:108, :, :]

    J_M1_1 = J_M1_tensor[:27, :, :]
    J_M1_2 = J_M1_tensor[27:54, :, :]
    J_M1_3 = J_M1_tensor[54:81, :, :]
    J_M1_4 = J_M1_tensor[81:108, :, :]

    return J_PMd_1, J_PMd_2, J_PMd_3, J_PMd_4, J_M1_1, J_M1_2, J_M1_3, J_M1_4


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
    # C_2 = np.cov(matrix.T)

    # Running SVD on the covariance matrix
    U,S_,V_T = np.linalg.svd(matrix, full_matrices=False)
    s_tot = np.sum(S_)
    frac = S_**2 / np.sum(S_**2)

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


def run_PCA (matrix, rank):
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
    C_2 = matrix.T @ matrix
    C_2 = C_2 / matrix.shape[0]
    # runs PCA
    U, S_, V_T = svd(C_2)

    # project the mean centered data onto these PCs to produce a rank k approximation
    proj = matrix @ U[:, :rank]
   
    return proj, U[:, :rank]


def mse_fun(true_values, predicted_values):
    """
    This function computes the mean squared error between the true values and the predicted values.

    Parameters:
        true_values: numpy array of true values
        predicted_values: numpy array of predicted values
    """
    return np.mean((true_values - predicted_values) ** 2)


def regress (train_N, train_M, lam):
    """
    This function performs ridge regression on the data matrix M with regularization parameter lam.


    Parameters:
        train_N: numpy array of shape [time-1, rank 2k] containing rank approximation of the neural data
        train_M: numpy array of shape [time-1, rank k] containing rank approximation of the muscle data
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
    W_hat = np.linalg.solve(C + (lam * I), train_N.T @ train_M)
   
    # compute the predicted values
    M_hat = train_N @ W_hat
   
    # compute R-squared values
    # per dimension
    R2_dims = 1 - np.var(train_M - M_hat, axis=0) / np.var(train_M, axis=0)

    # overall
    R2_total = 1 - np.sum((train_M - M_hat)**2) / np.sum((train_M - train_M.mean(axis=0))**2)

    # MSE
    MSE_lam = mse_fun(train_M, M_hat)

    # RMSE
    RMSE_lam = np.sqrt(MSE_lam)

    return W_hat, M_hat, R2_total, RMSE_lam, MSE_lam


def best_lam(neu_lam, mus_lam, time_bins):
    """
    This function takes in the training data and will compute the best lambda value for ridge regression using cross-validation. It will return the best lambda
    value and the mean squared error for that lambda.


    Parameters:
        neu_lam: a 2D numpy array of shape [ct, rank] which is the projection onto the first rank PCs for neural data
        mus_lam: a 2D numpy array of shape [ct, rank] which is the projection onto the first rank PCs for muscle data
        time_bins: the number of time bins used for each condition


    Returns:
        best_lambda: the best lambda value found during cross-validation
        min_mse: the root mean squared error of the subset of data using the best lambda
        min_rmse: the mean squared error of the subset of data using the best lambda
        mse_vals: all mean squared error values for each tested value of lambda
        rmse_vals: all root mean squared error values for each tested value of lambda


    """

    # shape data into a tensor
    conds = int(neu_lam.shape[0] / time_bins)
    neu_tensor = shape_tensor(neu_lam, conds)
    mus_tensor = shape_tensor(mus_lam, conds)

    # set up folds and random conditions
    K = min(5, conds)
    cond_idx = np.arange(conds)
    # np.random.seed(42)
    np.random.shuffle(cond_idx)
    folds = np.array_split(cond_idx, K)

    # Define a range of lambda values to test and initialize arrays
    lambdas = np.logspace(-4, 5, 40)
    mse_vals = []
    rmse_vals = []

    for lam in lambdas:

        fold_mse = []

        for k in range(K):

            val_idx = folds[k]
            train_idx = np.hstack([folds[i] for i in range(K) if i != k])

            #take training data out
            neu_train = shape_matrix(neu_tensor[train_idx])
            mus_train = shape_matrix(mus_tensor[train_idx])

            # take testing data out
            neu_val = shape_matrix(neu_tensor[val_idx])
            mus_val = shape_matrix(mus_tensor[val_idx])

            # recover W_hat
            W_hat, _, _, _, _ = regress(neu_train, mus_train, lam)

            # create estimation for test values and recover MSE
            mus_pred = neu_val @ W_hat
            mse = np.mean((mus_val - mus_pred)**2)

            fold_mse.append(mse)

        mean_mse = np.mean(fold_mse)

        mse_vals.append(mean_mse)
        rmse_vals.append(np.sqrt(mean_mse))

    #Identifying the best lambda
    best_idx = np.argmin(mse_vals)

    best_lambda = lambdas[best_idx]
    min_mse = mse_vals[best_idx]
    min_rmse = rmse_vals[best_idx]

    # Return the best lambda and its corresponding MSE        
    return best_lambda, mse_vals, rmse_vals


def simple_lam(N_train, M_train):
    """
    This function is using the scikit-learn package to perform cross validation on the samples. The metric is MSE. This is using leave one out cross validation.
    My caution with this is that I do not know what "one" was left out.


    Parameters:
        N_train: a 2D numpy array of shape [ct, rank] which is the projection onto the first rank PCs for neural data
        M_train: a 2D numpy array of shape [ct, rank] which is the projection onto the first rank PCs for muscle data


    Returns:
        best_lambda: the best lambda value found during cross-validation
        cv_results: the stored mean squared error for all tested lambdas
    """

     # Define a range of lambda values to test
    lambdas = np.logspace(-3, 3, 25)

    # Initialize the RidgeCV model with the lambda values, this will minimize MSE, and is data centered
    model_cv = RidgeCV(alphas=lambdas, scoring='neg_mean_squared_error', store_cv_results=True, fit_intercept = False)

    # Fit the model (the optimal alpha is found automatically during this step)
    model_cv.fit(N_train, M_train)
    cv_results = model_cv.cv_results_

    # The optimal alpha value can be accessed via the alpha_ attribute
    best_lambda = model_cv.alpha_
    print(">>> best_lam returning:", best_lambda)
    return best_lambda, cv_results


def r_regress (N_tilde, M_tilde, num_bins, J, PMd, cv = True):
    """
    Takes in M and N matrices and runs ridge regression on these matrices projected onto their first N_dim and M_dim PCs
    to generate a weight matrix (W) so that M_hat = N W. Also calculates R squared values.


    Parameters:
        M_tilde: This is a 2D matrix [conditions x time bins, muscle/neuron readouts] which is dependent on N
        N_tilde: This is a 2D matrix [conditions x time bins, neurons]
        num_bins: how many time bins are in each trial
        J: boolean identifier for monkey J or N
        PMd: boolean identified for PMd to M1 data or neurons to muscles
        cv: boolean which decides if the code should use the best_lam function (True) or the simple_lam function for cross-validation


    Returns:
        W: weight matrix found using ridge regression
        R2_total: the R squared value for the whole matrix 
        R2_dims: one value of R squared for every column of M_hat
        MSE_all: mean squared error of the reconstruction of M_tilde with the multiplication N_tilde and W
        RMSE_all: the square root of MSE_all
   
    """
    # conditions in the sample
    conds = int(N_tilde.shape[0] / num_bins)
   
    # listing and shuffling all possible indexes
    all_idx = np.arange(conds)
    # np.random.seed(42)
    np.random.shuffle(all_idx)

    # calculate the sizes of each set
    split = int((conds * 0.2))    # 20% for testing
    
    train_idx = all_idx[split:]
    test_idx = all_idx[:split]

    print(train_idx)

    # shaping back into a tensor
    neu_tensor = shape_tensor(N_tilde, conds)
    mus_tensor = shape_tensor(M_tilde, conds)

    # isolating the data as they've been split above        
    neu_test_tens = neu_tensor[test_idx, :, :]        # 20% for testing
    mus_test_tens = mus_tensor[test_idx, :, :]        # 20% for testing
    neu_train_tens = neu_tensor[train_idx, :, :]      # 80% for training
    mus_train_tens = mus_tensor[train_idx, :, :]      # 80% for training

    # reshaping into matrix        
    neu_test_mat = shape_matrix(neu_test_tens)        
    mus_test_mat = shape_matrix(mus_test_tens)        
    neu_train_mat = shape_matrix(neu_train_tens)      
    mus_train_mat = shape_matrix(mus_train_tens)

    # mean centering
    # neu_train_mat -= np.mean(neu_train_mat, axis=0, keepdims=True)
    # mus_train_mat -= np.mean(mus_train_mat, axis=0, keepdims=True)
    # neu_test_mat -= np.mean(neu_train_mat, axis=0, keepdims=True)
    # mus_test_mat -= np.mean(mus_train_mat, axis=0, keepdims=True)      

    # neu_train_mat = N_tilde
    # mus_train_mat = M_tilde 

    # Calling best lambda
    if cv:
        lam, _, _ = best_lam(neu_train_mat, mus_train_mat, num_bins)
    else:
        lam, _ = simple_lam(neu_train_mat, mus_train_mat)

    # setting up for regression
    neu_train_cov = neu_train_mat.T @ neu_train_mat
    I = np.identity(neu_train_cov.shape[0])

    # if J and not PMd:
    #     lam = 100
   
    # elif not J and PMd:
    #     lam = 58.780160722749116
    print(">>> best_lam returning:", lam)
    
    # retrieving the weights matrix for M_tilde = W N_tilde and the sum of squares regression using the training data
    W = np.linalg.solve(neu_train_cov + (lam * I), neu_train_mat.T @ mus_train_mat)

    # calculating the M_hat by multiplying neu_test_mat and W from above
    M_test_hat = neu_test_mat @ W
    M_hat = N_tilde @ W

    # calculating R squared for each column of M_tilde
    # per dimension
    R2_dims = 1 - np.var(M_tilde - M_hat, axis=0) / np.var(M_tilde, axis=0)

    # overall
    R2_total = 1 - np.sum((M_tilde - M_hat)**2) / np.sum((M_tilde - M_tilde.mean(axis=0))**2)
   
    # calcualting mean squared error of the reconstruction of mus_test_mat with the multiplication of neu_test_mat and W
    MSE_test = mse_fun(mus_test_mat, M_test_hat)
    RMSE_test = np.sqrt(MSE_test)

    # RMSE and MSE for whole dataset
    MSE_all = mse_fun(M_tilde, M_hat)
    RMSE_all = np.sqrt(MSE_all)
   
    return W, R2_total, R2_dims, MSE_all, RMSE_all
   
def scaling (tensor, tuning = False):
    """
    Takes in a tensor of shape [conditions, neurons, time bins] and scales it between 0 and 1. Then returns a tall and skinny 2D matrix of
    shape [conditions x time bins, neurons].


    Parameters:
        tensor: a 3D tensor of shape [conditions, neurons, time bins]
        tuning: across time x neuron scaling


    Returns:
        norm_matrix: a 2D version of tensor (shaped [conditions x time bins, neurons]) which is scaled between 0 and 1
    """
    check = tensor.shape[0]

    # if tuning:

    #     """
    #     Full preprocessing:
    #     1. Soft normalize (Kaufman-style)
    #     2. Subtract across-condition mean at each timepoint
    #     """
    #     epsilon = 5
    #     # --- soft normalization ---
    #     max_val = np.max(tensor, axis=(0,2), keepdims=True)
    #     min_val = np.min(tensor, axis=(0,2), keepdims=True)
    #     range_val = max_val - min_val
    #     tensor = tensor / (range_val + epsilon)
       
    #     # --- condition-wise centering ---
    #     mean_across_cond = np.mean(tensor, axis=0, keepdims=True)
    #     tensor = tensor - mean_across_cond
       
    #     # reshape into matrix
    #     matrix = shape_matrix(tensor)

    #     return matrix
   
    # else:

    if check < 300:
        new_matrix = shape_matrix(tensor)
    
    else:
        new_matrix = tensor

    # trying other form of scaling
    stand = np.std(new_matrix, axis = 0)

    standardized = np.zeros_like(new_matrix)
    norm_matrix = np.zeros_like(new_matrix)
    

    # columns max and min
    col_max = np.amax(new_matrix, axis = 0)
    col_min = np.amin(new_matrix, axis = 0)

    # Z-scoring
    mean = np.mean(new_matrix, axis=0)
    std = np.std(new_matrix, axis=0)
    std[std == 0] = 1
    standardized = (new_matrix - mean) / std

    for i in range(norm_matrix.shape[1]):
        norm_matrix[:, i] = (new_matrix[:, i]) / (col_max[i] - col_min[i])
        norm_matrix[:, i] = norm_matrix[:,i] - np.mean(norm_matrix[:,i])
    
    return(norm_matrix)
   
def slice (tensor_N): 
    N_prep_start = 30
    N_prep_end = 80   # 81 because it will get spliced off otherwise

    # retrieving dataset specifications
    J, PMd = ident(tensor_N)

    # altering movement periods depending on dataset
    if J:
        N_move_start = 150
        N_move_end = 216
    else:
        N_move_start = 142
        N_move_end = 208
        
    N_idx = np.r_[N_prep_start:N_prep_end, N_move_start:N_move_end]
    N_prep = tensor_N[:,:, N_prep_start:N_prep_end]
    N_move = tensor_N[:,:, N_move_start:N_move_end]

    N_prep_mat = shape_matrix(N_prep)
    N_move_mat = shape_matrix(N_move)

    prep_good = np.zeros(N_prep_mat.shape[1])
    move_good = np.zeros(N_move_mat.shape[1])
    all_good = []
    print(all_good)
    print(all_good.dtype)

    mean_prep = np.mean(N_prep_mat, axis = 0) 
    mean_move = np.mean(N_move_mat, axis = 0) 

    for i in range (N_prep_mat.shape[1]):
        check = False

        if  mean_prep[i] > 3: 
            prep_good[i] = 1 
            check = True
        else: 
            prep_good[i] = 0
        
        if  mean_move[i] > 5: 
            move_good[i] = 1 
            if check: 
                all_good.append(i)
        else: 
            move_good[i] = 0
        

    N_idx = np.r_[(all_good)]
    sliced_tensor = tensor_N[:, N_idx, :]
    return tensor_N

def fig_3_cut_t(tensor, dimensions):
    """
    This function takes in the interpPSTH 3D tensor [conditions, neurons, time bins] and will create figure 3 from the Churchland et al. 2012 paper.


    Parameters:
        tensor: must be an interpPSTH array which has the shape [conditions, neurons, time bins]
        dimensions: the number of dimensions to project onto (should be between 6 and 10)
    """
    # retrieving dataset specifications
    J, _ = ident(tensor)
    # tensor = slice(tensor)

    # scaling, mean centering, and arranging the tensor into a matrix
    N_matrix, _  = time_cut(tensor)
    conditions, _, _ = tensor.shape
    new_bins = int(N_matrix.shape[0] / conditions)

    # gathering the left vectors and projecting the N_matrix onto them
    proj, _ = run_PCA(N_matrix, dimensions)
   
    # returning the scaled, mean centered, and time cut matrix into a tensor  
    scaled_tensor = shape_tensor(proj, conditions, new_bins)

    # starting the figures
    fig, axs = plt.subplots(dimensions - 1, dimensions -1, figsize=(12, 6))
    axs = axs.flatten()
    c = 0
   
    for i in range(dimensions - 1):
           
        for k in range(dimensions):
           
            if k != i:          
                for j in range(conditions):
                    current_cond = scaled_tensor[j, :, :]
                    current_cond = np.squeeze(current_cond)
                    dim1 = current_cond[i, :]
                    dim2 = current_cond[k, :]


                    if (i < dimensions - 1):
                        axs[c].plot(dim1[:51], dim2[:51], '-', color='blue', label='Preparatory')
                        axs[c].plot(dim1[51], dim2[51], 'o', color='gray', label='Go')
                        axs[c].plot(dim1[52:117], dim2[52:117], '-', color='green', label='Movement')
                        axs[c].plot(dim1[117], dim2[117], 'o', color='red', label='Movement')


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
    d1 -= 1
    d2 -= 1
        # retrieving dataset specifications
    J, _ = ident(tensor)
    # tensor = slice(tensor)
    

    # scaling, mean centering, and arranging the tensor into a matrix
    N_matrix, _  = time_cut(tensor)
    conditions, _, _ = tensor.shape
    new_bins = int(N_matrix.shape[0] / conditions)

    # gathering the left vectors and projecting the N_matrix onto them
    proj, _ = run_PCA(N_matrix, 2)
   
    # returning the scaled, mean centered, and time cut matrix into a tensor  
    scaled_tensor = shape_tensor(proj, conditions, new_bins)
   
    for i in range(conditions):
        current_cond = scaled_tensor[i, :, :]
        current_cond = np.squeeze(current_cond)
        dim1 = current_cond[d1, :]
        dim2 = current_cond[d2, :]

        if i == 0:
            plt.plot(dim1[:51], dim2[:51], '-', color='blue', label='Preparatory')
            plt.plot(dim1[51], dim2[51], 'o', color='gray', label='Go')
            plt.plot(dim1[52:117], dim2[52:117], '-', color='green', label='Movement')
            plt.annotate(
                'End',
                xy=(dim1[117], dim2[117]),
                xytext=(dim1[116], dim2[116]),
                arrowprops=dict(arrowstyle='-|>', color='green', lw=2),
                color='green'
            )
        else:
            plt.plot(dim1[:51], dim2[:51], '-', color='blue')
            plt.plot(dim1[51], dim2[51], 'o', color='gray')
            plt.plot(dim1[52:117], dim2[52:117], '-', color='green')
            plt.annotate(
                '',
                xy=(dim1[117], dim2[117]),
                xytext=(dim1[116], dim2[116]),
                arrowprops=dict(arrowstyle='-|>', color='green', lw=2),
            )


    ax = plt.gca()
    plot_cov_ellipse(
        prep_x, prep_y, ax,
        n_std=2,
        edgecolor='red',
        facecolor='none',
        linewidth=2,
        label='Prep (2 s.d.)'
    )

    plt.xlabel(f"Dimension {d1 + 1}")
    plt.ylabel(f"Dimension {d2 + 1}")

    if J:
        plt.legend(loc = 3)
        plt.title("Monkey J Neural Projection")

    else:
        plt.legend(loc = 2)
        plt.title("Monkey N Neural Projection")


    plt.tight_layout()
    plt.show()


def time_shift(tensor_N, tensor_M, scale = False, fig4 = False, J_PMd = False):
    """
    This function will both splice the data based on critical time events referenced in the paper. This is
    necessary before PCA or anything can be run on the data


    Parameters:
        tensor_N: This is the inter_PSTH for the N matrix in the equation M = WN
        tensor_M: this is the inter_PSTH for the M matrix in the equation M = WN
        PMd: boolean which tells time shift whether or not to add a 50ms delay (only should be done in the case of muscle data)
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
    # goCue (1550ms for J, 1470ms for N) and 600ms after. Motor activity is shifted 50ms later to account for signalling delay and only includes movement
    # data
    # cutting the N tensor with the times in preparatory period and movement period
   
    N_prep_start = 30
    N_prep_end = 81     # 81 because it will get spliced off otherwise

    # retrieving dataset specifications

    J, PMd = ident(tensor_N)
    if J_PMd: 
            PMd = True
    # altering movement periods depending on dataset
    if J:
        N_move_start = 150
        N_move_end = 216
    else:
        N_move_start = 142
        N_move_end = 208
    

    # retrieving specific indexes for figure 4
    if fig4:
        N_prep_start = 0
        N_prep_end = 81
        if J:                          
            N_move_start = 125
            N_move_end = 216
        else:
            N_move_start = 117
            N_move_end = 208
    
    N_idx = np.r_[N_prep_start:N_prep_end, N_move_start:N_move_end]
    N_cut = tensor_N[:,:, N_idx]
    
    # isolates movement data needed for the regression to find W tilde and tuning
    N_move = tensor_N[:,:, N_move_start:N_move_end]

    # cutting the M tensor with the times in movement period, depending on if it maps to muscles or not
    if PMd:
        M_move_start = N_move_start
        M_move_end = N_move_end
    else:
        M_move_start = N_move_start + 5
        M_move_end = N_move_end + 5
    M_idx = np.r_[M_move_start:M_move_end]
    M_move = tensor_M[:,:, M_idx]


    N_cut_scale = scaling(N_cut, tuning = scale)
    N_move_scale = scaling(N_move, tuning = scale)
    M_move_scale = scaling(M_move, tuning = scale)

    return N_cut_scale, N_move_scale, M_move_scale


def time_cut (tensor):
    """
    Will splice out the times for preparatory activity and motor activity, used for figure 3. Includes the goCue


    Parameters:
        tensor: inter_PSTH tensor [conditions, neurons, time bins]


    Returns:
        N_mean_scaled: 2D inter_PSTH matrix with only time bins during preparatory activity, go cue, and movement that has been scaled and centered
        N_m_mean_scaled: 2D inter_PSTH matrix with only time bins during movement that has been scaled and centered
    """

    # retrieving dataset specifications
    J, _ = ident(tensor)

    # altering movement periods depending on dataset and getting indexes for time cuts (includes go cue needed in figure 3)
    if J:
        N_idx = np.r_[30:81, 120, 150:216]
        N_move = np.r_[150:216]
    else:
        N_idx = np.r_[30:81, 118, 142:208]
        N_move = np.r_[142:208]
   
    # splicing data
    N_tens = tensor[:,:, N_idx]
    N_m_tens = tensor[:,:, N_move]

    # scaling and mean centering it into a matrix
    N_scale = scaling(N_tens, False)
    N_m_scale = scaling (N_m_tens, False)

    return N_scale, N_m_scale


def fig_4 (tensor_N, tensor_M, dimensions = 6, plot = False, basis = 0, cv = True, basis_2 = 0, J_sect = 2):
    """
    Performs regression needed for figure 4.


    Parameters:
        tensor_N: This is the inter_PSTH tensor [conditions, neurons, time] for the N matrix in the equation M = WN
        tensor_M: this is the inter_PSTH tensor [conditions, muscles, time] for the M matrix in the equation M = WN
        PMd: boolean which tells time shift whether or not to add a 50ms delay (only should be done in the case of muscle data)
        dimensions: the amount of dimensions matrix N should be reduced to
        plot: boolean which will call fig_4_plot if True
        basis: parameter for fig_4_plot
        J_sect: start from 1-4, tells which quadrant of the data it will use 
       


    Returns:
        W: This is W tilde, the low rank approximation of the weights matrix
        M_hat: The reconstruction of M_tilde through multiplying N_tilde and W
        M_recon: M_hat projected onto the PCs of M which were used to produce M_tilde
        R_squared: array of the R squared values for each column of M_hat, in comparison to M_tilde
        MSE: The mean squared error of M_hat in comparison to M_tilde
    """

    # retrieving dataset specifications
    J, PMd = ident(tensor_N)
    print(J)
    print(PMd)
    if J and PMd:
        J_sect -= 1
        lower = J_sect*27
        upper = lower + 27
        tensor_N = tensor_N[lower:upper, :, :]
        tensor_M = tensor_M[lower:upper, :, :]

    # retrieving number of conditions
    cond = tensor_N.shape[0]

    # tensor_N = slice(tensor_N)

    # if PMd: 
        # tensor_M = slice(tensor_M)

    # scaling, mean centering, and involving only the time periods needed for regression (the movement)
    regress_N, move_N, regress_M = time_shift(tensor_N, tensor_M)
    time_ct = regress_M.shape [0]
    time_ct_neu = regress_N.shape [0]

    # retrieving data projected onto the first N_dim and M_dim PCs
    N_tilde,N_PCs = run_PCA(regress_N, dimensions)
    M_tilde,PCs = run_PCA(regress_M, int(dimensions/2))

    # how many time bins are included in the movement period
    time_bins = int(time_ct / cond)

    # how many time bins are included in the preparatory and movement period
    time_bins_pm = int(time_ct_neu / cond)

    # difference in bins = prep bins
    diff_bin = int((time_bins_pm - time_bins))
   
    # removing prep bins adn reshaping for ridge
    regress_N = shape_tensor(N_tilde, conditions = cond, time_bins = time_bins_pm)
    N_tens_spliced = regress_N[:,:, diff_bin:]
    regress_N_sp = shape_matrix(N_tens_spliced)

    # running through ridge regression
    W, R2_total, R2_dim, MSE_all, RMSE_all = r_regress(regress_N_sp, M_tilde, num_bins = time_bins, J = J, PMd = PMd, cv = cv)

    if plot:
        regress_N, _, _ = time_shift(tensor_N, tensor_M, fig4 = True)  # getting new regression N which includes more time points to match their graphs
        N_tilde = regress_N @ N_PCs  # projecting onto same PCs as earlier
        fig_4_plot(W, N_tilde, cond, dimensions, basis, J, basis_2)
    return W, R2_total, R2_dim, MSE_all, RMSE_all


def fig_4_plot (W, N_tilde, cond, dimensions, basis = 0, J = True, basis_2 = 0):
    '''
    Plot needed for figure 4.


    Parameters:
        W: This is W tilde, the low rank approximation of the weights matrix
        N_tilde: this is the low rank approximation of matrix N (includes prep and movement)
        cond: the number of conditions used
        basis: which of the three potent/null dimensions will be plotted
        J: tells if this is from monkey J or N
       
    Returns:
        plot of the neural activity in the potent and null space
    '''


    # running SVD on W to be able to get the null space of the matrix
    U, S_val, V = np.linalg.svd(W)
    S_val = np.diag(S_val)
    rank = int(dimensions/2)

    # potent and null space basis of W
    W_potent = U[:,:rank]
    W_null = U[:,rank:]

    # low rank neural data projected onto null and potent space of weights and scaling them
    N_potent =  N_tilde @ W_potent
    N_null = N_tilde @ W_null
    max = np.max(np.abs(np.concatenate([N_potent, N_null])))
    N_potent /= max
    N_null /=  max

    # setting up time for x axis
    prep_time = np.arange(0, 810, 10)
    if J:
        move_time = np.arange(1250, 2160, 10)
    else:
        move_time = np.arange(1170, 2080, 10)
   
    # setting up for loop
    time_bins = int(N_potent.shape[0] / cond)

    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(2, 1, figure=fig)

    # different limits on the axes depending on which dataset was given
    if J:
        bax1 = brokenaxes(xlims=((0, 800), (1250, 2170)), ylims=((-1.25, 1.25),), hspace=.05, subplot_spec=gs[0])
        bax2 = brokenaxes(xlims=((0, 800), (1250, 2170)), ylims=((-1.25, 1.25),), hspace=.05,  subplot_spec=gs[1])
    else:
        bax1 = brokenaxes(xlims=((0, 800), (1170, 2090)), ylims=((-1.25, 1.25),), hspace=.05, subplot_spec=gs[0])
        bax2 = brokenaxes(xlims=((0, 800), (1170, 2090)), ylims=((-1.25, 1.25),), hspace=.05,  subplot_spec=gs[1])  
    print("prep:", len(prep_time))
    print("move:", len(move_time))
    print("segment size:", time_bins)
    print("total N rows:", N_null.shape[0])
    print("expected per cond:", len(prep_time) + len(move_time))

    # labels for output null graph
    bax1.text(500, -1.1, "Test Epoch", ha='center')
    bax1.text(1800, -1.1, "Regression Epoch", ha='center')
    bax1.set_title(f"Output Null Dimension {basis + 1}")
    bax1.set_xlabel("Time in Trial")
    bax1.set_ylabel("Projection (a.u.)")
 
    y_line = -1.2  # slightly above lower y-limit

    # 300–800 (blue) for preparatory activity
    bax1.plot([300, 800], [y_line, y_line],
            color='blue', linewidth=4, solid_capstyle='butt')
    bax2.plot([300, 800], [y_line, y_line],
            color='blue', linewidth=4, solid_capstyle='butt')
    if J:
        # 1500–2170 (green) for regression epoch
        bax1.plot([1500, 2170], [y_line, y_line],
                color='green', linewidth=4, solid_capstyle='butt')
        bax2.plot([1500, 2170], [y_line, y_line],
                color='green', linewidth=4, solid_capstyle='butt')
       
    else:
            # 1420–2080 (green) for regression epoch
        bax1.plot([1420, 2080], [y_line, y_line],
                color='green', linewidth=4, solid_capstyle='butt')
        bax2.plot([1420, 2080], [y_line, y_line],
                color='green', linewidth=4, solid_capstyle='butt')

    print(J)
    # plotting the data for output null space
    for i in range(cond):
        start_prep = i* time_bins
        end_prep = start_prep + len(prep_time)
        end_move = end_prep + len(move_time)

        if i == 0:
            bax1.plot(prep_time, N_null[start_prep:end_prep, basis], '-', color='midnightblue', label = 'null',  linewidth = .5)
            bax1.plot(move_time, N_null[end_prep:end_move, basis], '-', color='midnightblue',  linewidth = .5)
        else:
            bax1.plot(prep_time, N_null[start_prep:end_prep, basis], '-', color='midnightblue',  linewidth = .5)
            bax1.plot(move_time, N_null[end_prep:end_move, basis], '-', color='midnightblue',  linewidth = .5)
   
    # labels for output potent graph
    bax2.text(500, -1.1, "Test Epoch", ha='center')
    bax2.text(1800, -1.1, "Regression Epoch", ha='center')
    bax2.set_title(f"Output Potent Dimension {basis_2 + 1}")
    bax2.set_xlabel("Time in Trial")
    bax2.set_ylabel("Projection (a.u.)")


    # plotting the data for output potent space
    for i in range(cond):
        start_prep = i* time_bins
        end_prep = start_prep + len(prep_time)
        end_move = end_prep + len(move_time)

        if i == 0:
            bax2.plot(prep_time, N_potent[start_prep:end_prep, basis_2], '-', color='darkmagenta', label = 'potent',  linewidth = .5)
            bax2.plot(move_time, N_potent[end_prep:end_move, basis_2], '-', color='darkmagenta',  linewidth = .5)
        else:
            bax2.plot(prep_time, N_potent[start_prep:end_prep, basis_2], '-', color='darkmagenta', linewidth = .5)
            bax2.plot(move_time, N_potent[end_prep:end_move, basis_2], '-', color='darkmagenta',  linewidth = .5)
   
    if J:
        # prep ticks
        bax1.axs[0].set_xticks([0, 400, 800])
        bax1.axs[0].set_xticklabels(['-400', 'targ', '400'])
        bax2.axs[0].set_xticks([0, 400, 800])
        bax2.axs[0].set_xticklabels(['-400', 'targ', '400'])

        # movement ticks
        bax1.axs[1].set_xticks([1250, 1550, 2170])
        bax1.axs[1].set_xticklabels(['-300', 'move', '600'])
        bax2.axs[1].set_xticks([1250, 1550, 2170])
        bax2.axs[1].set_xticklabels(['-300', 'move', '600'])
    else:
        # prep ticks
        bax1.axs[0].set_xticks([0, 400, 800])
        bax1.axs[0].set_xticklabels(['-400', 'targ', '400'])
        bax2.axs[0].set_xticks([0, 400, 800])
        bax2.axs[0].set_xticklabels(['-400', 'targ', '400'])

        # movement ticks
        bax1.axs[1].set_xticks([1170, 1470, 2090])
        bax1.axs[1].set_xticklabels(['-300', 'move', '600'])
        bax2.axs[1].set_xticks([1170, 1470, 2090])
        bax2.axs[1].set_xticklabels(['-300', 'move', '600'])
   
    # bax1.legend(loc = 2)
   
def tuning_rat (W_potent, W_null, neu_move, neu_prep, get_gamma = False, cond = 108):
    """
    Takes in the weights matrix from ridge regression and it's null space, as well as neural activity from the movement and preparatory period and computes the
    tuning ratio in two ways. The first returns it with using the sum of variance, and the second with the squared frobenius norm. Tuning ratio is computed as
    described in the methods section.


    Parameters:
        W_potent: the potent space of the weights matrix found with ridge regression
        W_null: the null space of the weights matrix found with ridge regression
        neu_move: dimensionally reduced neural matrix which contains time period 50ms before to 600ms after the movement starts
        neu_prep: dimensionally reduced neural matrix which contains time period 100ms before to 400ms after the target onset
   
    Returns:
        var_tuning: tuning ratio computed using the sum of variance
        frob_tuning: tuning ratio computed using the frobenius norm squared
    """
    # movement null and potent space for gamma
    N_null_move = neu_move @ W_null
    # N_nm_tensor = shape_tensor(N_null_move, cond)
    N_null_move = N_null_move - N_null_move.mean(axis=0)     # the other one to comment out
    # N_null_move = shape_matrix(N_nm_tensor)
    null_move_frob = np.linalg.norm(N_null_move)**2
    null_move_var = np.sum(np.var(N_null_move, axis=0))

    N_pot_move = neu_move @ W_potent
    # N_pm_tensor = shape_tensor(N_pot_move, cond)
    N_pot_move = N_pot_move - N_pot_move.mean(axis=0)     # the one to comment out
    # N_pot_move = shape_matrix(N_pm_tensor)
    pot_move_frob = np.linalg.norm(N_pot_move)**2
    pot_move_var = np.sum(np.var(N_pot_move, axis=0))
   
    # computing gamma which is a scaling factor
    gamma = null_move_var / pot_move_var
    gamma2 = null_move_frob / pot_move_frob

    # Null and potent projections of movement neural data
    N_null_prep = neu_prep @ W_null
    # N_np_tensor = shape_tensor(N_null_prep, cond)
    N_null_prep = N_null_prep - N_null_prep.mean(axis=0) 
    # N_null_prep = shape_matrix(N_np_tensor)       # subtract columns for variance
    null_prep_frob = np.linalg.norm(N_null_prep)**2
    null_prep_var = np.sum(np.var(N_null_prep, axis=0))

    N_pot_prep = neu_prep @ W_potent
    # N_pp_tensor = shape_tensor(N_pot_prep, cond)
    N_pot_prep = N_pot_prep - N_pot_prep.mean(axis=0) 
    # N_pot_prep = shape_matrix(N_pp_tensor)       # subtract columns for variance
    pot_prep_frob = np.linalg.norm(N_pot_prep)**2
    pot_prep_var = np.sum(np.var(N_pot_prep, axis=0))

    # tuning ratio
    var_tuning = (null_prep_var / pot_prep_var) * ( 1/ gamma )   # this is with using the sum of variance
    frob_tuning = (null_prep_frob / pot_prep_frob) * (1 / gamma2 )   # this is with using the frobenius norm and not variance on the movement data

    # fraction of prep in null space and potent space
    null_fraction = null_prep_var / (null_prep_var + pot_prep_var)
    pot_fraction  = pot_prep_var / (null_prep_var + pot_prep_var)
    if get_gamma:
        return gamma                                                           # RETURNING GAMMA 2

    print("1/Gamma: ", 1/gamma)
    print("1/Gamma2: ", 1/gamma2)
    print("Move null/pot:", null_move_var / pot_move_var)
    print("potent move variance: ", pot_move_var)
    print("null move variance: ", null_move_var)
    print("null prep variance:", null_prep_var)
    print("potent prep variance: ", pot_prep_var)
    print("Tuning with variance: ", var_tuning)
    print("Tuning with frob: ", frob_tuning)
    print("Prep fraction: ",  null_fraction)
    return var_tuning, frob_tuning, null_fraction, pot_fraction


def tuning_setup (N_tilde_move, M_tilde, N_tilde_prep, dims, time_bins, J, PMd, rep = 0, time = False):
    """
    Takes in two tensors and processes them to get the tuning ratio.

    Parameters:
        tensor_N: tensor which has either neural data or PMd data
        tensor_M: tensor which has either muscle data or M1 data
        PMd: boolean which tells time shift whether or not to add a 50ms delay (only should be done in the case of muscle data)
        cv: choosing method of cross-validation, True = method called best lambda
        rep: how many repeats it should perform
   
    Returns:
        var_tuning:
        frob_tuning:
    """
    cond = int(N_tilde_move.shape[0] / time_bins)

    var_tuning = []
    frob_tuning = []
    null_frac = []
    pot_frac = []

    for i in range(rep + 1):
        # computing W
        W1,  _, _, _, _ = r_regress(N_tilde_move, M_tilde, num_bins = time_bins, J = J, PMd = PMd)
        U, S_val, V = np.linalg.svd(W1, full_matrices = True)
        rank = int(dims/2)

        # potent and null space basis of W
        W_potent = U[:,:rank]
        W_null = U[:,rank:]

        if time:
            gamma = tuning_rat(W_potent, W_null, N_tilde_move, N_tilde_prep, get_gamma = True, cond = cond)
            return W_potent, W_null, gamma

        var_tuning_i, frob_tuning_i, null_frac_i, pot_frac_i = tuning_rat(W_potent, W_null, N_tilde_move, N_tilde_prep, cond = cond)
        var_tuning.append(var_tuning_i)
        frob_tuning.append(frob_tuning_i)
        null_frac.append(null_frac_i)
        pot_frac.append(pot_frac_i)
    return var_tuning, frob_tuning, null_frac, pot_frac


def tuning_mult (tensor_N, tensor_M, dims, plot = False, rep = 1):
    """
    Function which takes two tensors, performs reduced rank regression with the set of dimensions, and will plot the proportion of preparatory activity occupying the
    null space and potent space, as well as have the tuning ratio above it. The regression can be repeated multiple times for one set of dimension and the tuning
    ratios will be averaged, as well as the proportions mentioned before. Can also just return the values mentioned.


    Parameters:
        tensor_N: This is the inter_PSTH tensor [conditions, neurons, time] for the N matrix in the equation M = WN
        tensor_M: this is the inter_PSTH tensor [conditions, muscles, time] for the M matrix in the equation M = WN
        dims: the amount of dimensions matrix N should be reduced to
        PMd: boolean which tells time shift whether or not to add a 50ms delay (only should be done in the case of muscle data)
        plot: boolean which will form a plot if True
        rep: the number of times the regression will be repeated, then tuning ratios from these will be averaged together
        cv: boolean selecting RidgeCV cross validation (False for that package)
   
    Returns:
        var_tuning_means: an array which has one value for the average tuning ratio based on the repeats specified for one set of dimensions, used variance
        frob_tuning_means: an array which has one value for the average tuning ratio based on the repeats specified for one set of dimensions, used frobenius norm
        null_frac_means: an array which has values for the proportion of preparatory activity occupying the null space (1 value for each set of dimensions)
        pot_frac_means: an array which has values for the proportion of preparatory activity occupying the potent space (1 value for each set of dimensions)
    """

    # making sure this is the correct type of object for the for loop
    if type(dims) == int:
        dims = np.array([dims])

    # retrieving dataset specifications
    J, PMd = ident(tensor_N)

    if J and PMd: 
        J_rep = 4
    
        # initializing arrays to hold the average values for each set of dimensions
        var_tuning_means_ext = []
        frob_tuning_means_ext = []
        null_frac_means_ext = []
        pot_frac_means_ext = []
            
    else: 
        J_rep = 1
    initial = 0

    J_PMd = False
    

    for dim in dims: 

        for i in range (J_rep): 
            if J and PMd:
                J_PMd = True
                finish = i + 27
                tensor_N = tensor_N[i:finish, :, :]
                tensor_M = tensor_M[i:finish, :, :]

            # tensor_N = slice(tensor_N)
            # if PMd: 
                # tensor_M = slice(tensor_M)

            # setting up needed shape specifics
            cond, _, _ = tensor_N.shape

            # initializing arrays to hold the average values for each set of dimensions
            var_tuning_means = []
            frob_tuning_means = []
            null_frac_means = []
            pot_frac_means = []

            
            regress_N, _, regress_M = time_shift(tensor_N, tensor_M, J_PMd = J_PMd)          # normal range matrix for regression
            N_tilde, _ = run_PCA(regress_N, dim)
            M_tilde, _ = run_PCA(regress_M, int(dim/2))

            # lengths of conditions x time, regress M only has movement, whereas regress_N has prep and movement
            time_ct = regress_M.shape [0]
            time_ct_neu = regress_N.shape [0]

            # how many time bins are included in the movement period for a single condition
            time_bins = int(time_ct / cond)

            # how many time bins are included in the preparatory and movement period per condition
            time_bins_pm = int(time_ct_neu / cond)

            # difference in bins = just prep bins ie where the movement period starts for each condition
            diff_bin = int((time_bins_pm - time_bins))

            # isolating the preparatory and movement bins
            N_tilde_tens = shape_tensor(N_tilde, cond, time_bins_pm)
            N_tilde_tens_move = N_tilde_tens[:,:,diff_bin:]
            N_tilde_tens_prep = N_tilde_tens[:,:,:diff_bin]

            # reshape into matrices for tuning computation
            N_tilde_move = shape_matrix(N_tilde_tens_move)
            N_tilde_prep = shape_matrix(N_tilde_tens_prep)

            # retrieving tuning values and null and potent fraction for preparatory activity for each set of dimensionally reduced regression
            var_tuning, frob_tuning, null_frac, pot_frac = tuning_setup(N_tilde_move, M_tilde, N_tilde_prep, dims = dim, time_bins = time_bins, J = J, PMd = PMd, rep = rep)
            var_tuning_means.append(np.mean(var_tuning))
            frob_tuning_means.append(np.mean(frob_tuning))
            null_frac_means.append(np.mean(null_frac))
            pot_frac_means.append(np.mean(pot_frac))
        
        # J_PMd analysis needs to be average of the 4 subsets of data, so per dimension these four subsets are averaged 
        if J and PMd: 
            var_tuning_means_ext.append(np.mean(var_tuning_means))
            frob_tuning_means_ext.append(np.mean(frob_tuning_means))
            null_frac_means_ext.append(np.mean(null_frac_means))
            pot_frac_means_ext.append(np.mean(pot_frac_means))
            var_tuning_means = var_tuning_means_ext
            frob_tuning_means = frob_tuning_means_ext
            null_frac_means = null_frac_means_ext
            pot_frac_means = pot_frac_means_ext
            
    # plotting
    if plot:
       
        # Example data
       
        null_prop = np.array(null_frac_means)
        potent_prop = 1 - null_prop        # ensures they sum to 1
        print(null_prop)
        print(pot_frac_means)

        x = np.arange(len(dims))           # group positions
        width = 0.15                       # bar width

        fig, ax = plt.subplots(figsize=(4, 5))

        # Null bars (all same color)
        ax.bar(x - width/2, null_prop, width, label="Null", color='midnightblue')

        # Potent bars (all same color)
        ax.bar(x + width/2, potent_prop, width, label="Potent", color='darkmagenta')

        for i in range(len(dims)):
            ax.text(
            x[i],                      # center of the two bars
            max(null_prop[i], potent_prop[i]) + 0.05,  # above taller bar
            f"{var_tuning_means[i]:.2f}",     # <-- put whatever value you want here
            ha='center',
            va='bottom',
            fontsize=9
                )      
        ax.set_xticks(x)
        ax.set_xticklabels(dims)
        ax.set_xlabel("Number of Dimensions")
        ax.set_ylabel("Fraction of Preparatory Tuning")
        ax.set_ylim(0, 1)
        ax.legend()
        if J:
            J_text = "J"
        else:
            J_text = "N"
        if PMd:
            an_text = "PMd to M1"
        else:
            an_text = "Neurons to Muscles"
        title_text = f"Monkey {J_text} {an_text} Tuning Ratio"


        ax.set_title(title_text)
        plt.tight_layout()
        plt.show()

    else:
        return var_tuning_means, frob_tuning_means, null_frac_means, pot_frac_means
   
def sup_tuning (tensor_N, tensor_M, dims = 6, fig_4D = False):

    # retrieving dataset specifications
    J, PMd = ident(tensor_N)

    if J and PMd: 
        J_rep = 4
        J_PMd = True
    else: 
        J_rep = 1
        J_PMd = False

    V_null_ext = []
    V_pot_ext = []
    var_total_ext  = []
    var_null_ext   = []
    var_potent_ext = []

    frac_null_ext  = []
    frac_potent_ext = []

    for i in range (J_rep): 
        if J and PMd:
            finish = i + 27
            tensor_N = tensor_N[i:finish, :, :]
            tensor_M = tensor_M[i:finish, :, :]
            

        # tensor_N = slice(tensor_N)

        # if PMd: 
        #     tensor_M = slice(tensor_M)

        # getting weights matrix for potent and null space
        cond, _, fin_time = tensor_N.shape
        N_fig4, _, _ = time_shift(tensor_N, tensor_M, fig4 = True, J_PMd = J_PMd)     # elongated matrix for projection later
        regress_N, _, regress_M = time_shift(tensor_N, tensor_M, J_PMd = J_PMd)          # normal range matrix for regression
        N_tilde, PCs = run_PCA(regress_N, dims)
        M_tilde, _ = run_PCA(regress_M, int(dims/2))

        time_ct = regress_M.shape [0]
        time_ct_neu = regress_N.shape [0]

        # how many time bins are included in the movement period
        time_bins = int(time_ct / cond)

        # how many time bins are included in the preparatory and movement period
        time_bins_pm = int(time_ct_neu / cond)

        # difference in bins = just prep bins
        diff_bin = int((time_bins_pm - time_bins))

        # isolating the preparatory and movement bins
        N_tilde_tens = shape_tensor(N_tilde, cond, time_bins_pm)
        N_tilde_tens_move = N_tilde_tens[:,:,diff_bin:]
        N_tilde_tens_prep = N_tilde_tens[:,:,:diff_bin]

        # reshape into matrices for tuning computation
        N_tilde_move = shape_matrix(N_tilde_tens_move)
        N_tilde_prep = shape_matrix(N_tilde_tens_prep)

        # recovering the W_potent and W_null
        W_potent, W_null, gamma = tuning_setup(N_tilde_move, M_tilde, N_tilde_prep, dims, time_bins = time_bins, J = J, PMd = PMd, time = True)
    
        # projecting the expanded range onto the PCs recovered from the normal range
        N_tilde_full = N_fig4 @ PCs

        # projecting the neural activity of 400ms before and after target and 300ms before and 800ms after move starts onto the potent and null space of the weights matrix
        N_potent = N_tilde_full @ W_potent
        N_null = N_tilde_full @ W_null

        # mean centering
        N_potent = N_potent - np.mean(N_potent, axis = 0)
        N_null = N_null - np.mean(N_null, axis = 0)

        # reshaping into a tensor
        pot_tensor = shape_tensor(N_potent, cond)
        null_tensor = shape_tensor(N_null, cond)
        _, _, time = pot_tensor.shape

        # initializing array for holding the variance
        V_pot = np.zeros(time)
        V_null = np.zeros(time)
        
        var_total  = np.sum(np.var(N_tilde_full, axis=0))
        var_null   = np.sum(np.var(N_null, axis=0))
        var_potent = np.sum(np.var(N_potent, axis=0))

        frac_null   = var_null / var_total
        frac_potent = var_potent / var_total
        

        # goes through all time steps and pulls all conditions
        for t in range (time):
            X_pot  = pot_tensor[:, :, t]
            X_null = null_tensor[:, :, t]
            
        # variance 
            V_null[t] = np.sum(np.var(X_null, axis=0))
            V_pot[t]  = np.sum(np.var(X_pot, axis=0))

        
            # # squaring and adding values and dividing by condition numbers to compute variance
            # V_null[t] = np.sum(X_null**2) / (cond * dims)
            # V_pot[t]  = np.sum(X_pot**2)  / (cond * dims)
    
        V_null = (1/gamma) * V_null
        # V_pot = (1/gamma) * V_pot
        
        # appending if J_PMd
        if J_PMd:
            V_null_ext.append(V_null)
            V_pot_ext.append(V_pot)
            var_total_ext.append(var_total)
            var_null_ext.append(var_null)
            var_potent_ext.append(var_potent)
            frac_null_ext.append(frac_null)
            frac_potent_ext.append(frac_potent)



    # averaging 
        if J_PMd: 
            V_null_array = np.vstack(V_null_ext)
            V_null = np.mean(V_null_array, axis = 0)

            V_pot_array = np.vstack(V_pot_ext)
            V_pot = np.mean(V_pot_array, axis = 0)

            var_total = np.mean(var_total_ext)
            var_null = np.mean(var_null_ext)
            var_potent = np.mean(var_potent_ext)
            frac_null = np.mean(frac_null_ext)
            frac_potent = np.mean(frac_potent_ext)


    # print
    print("frac null: ", frac_null)
    print("frac potent: ", frac_potent)

    # initializing figure parameters
    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(1, 1, figure=fig)

    # time for plotting x axis and indexes needed for correct slicing
    prep_time = np.arange(0, 810, 10)
    prep_idx = np.arange(81)
    move_idx_start = len(prep_idx)
    move_end_4D = move_idx_start + 30

    if fig_4D:
        max = np.max(np.abs(np.concatenate([V_null[:move_end_4D], V_pot[:move_end_4D]])))
    else:
        max = np.max(np.abs(np.concatenate([V_null, V_pot])))
       
    # different limits on the axes depending on which dataset was given
    if J:
        J_text = "J"
        if fig_4D:
            move_time = np.arange(-300, 0, 10)
            bax1 = brokenaxes(
             xlims=((0, 800), (-300, 0)),
             ylims=((0, max + .2),),
             hspace=.05,
             subplot_spec=gs[0])
        else:
            bax1 = brokenaxes(xlims=((0, 800), (1250, 2170)), ylims=((0, max + .2),), hspace=.05, subplot_spec=gs[0])
            move_time = np.arange(1250, 2160, 10)
       
    else:
        J_text = "N"
        if fig_4D:
           move_time = np.arange(-300, 0, 10)
           bax1 = brokenaxes(
            xlims=((0, 800), (-300, 0)),
            ylims=((0, max + .2),),
            hspace=.05,
            subplot_spec=gs[0])
        else:
            bax1 = brokenaxes(xlims=((0, 800), (1170, 2090)), ylims=((0, max + .2),), hspace=.05, subplot_spec=gs[0])
            move_time = np.arange(1170, 2080, 10)
           
    if fig_4D:
        # plotting data
        move_end = move_idx_start + 30
        bax1.plot(prep_time, V_null[prep_idx], '-', color='midnightblue', label = 'null', linewidth = 1)
        bax1.plot(move_time, V_null[move_idx_start:move_end], '-', color='midnightblue',  linewidth = 1)
        bax1.plot(prep_time, V_pot[prep_idx], '-', color='darkmagenta', label = 'potent',  linewidth = 1)
        bax1.plot(move_time, V_pot[move_idx_start:move_end], '-', color='darkmagenta',  linewidth = 1)
    else:
    # plotting data
        bax1.plot(prep_time, V_null[prep_idx], '-', color='midnightblue', label = 'null', linewidth = 1)
        bax1.plot(move_time, V_null[move_idx_start:], '-', color='midnightblue',  linewidth = 1)
        bax1.plot(prep_time, V_pot[prep_idx], '-', color='darkmagenta', label = 'potent',  linewidth = 1)
        bax1.plot(move_time, V_pot[move_idx_start:], '-', color='darkmagenta',  linewidth = 1)

    if J:
        # preparatory ticks
        bax1.axs[0].set_xticks([0, 400, 800])
        bax1.axs[0].set_xticklabels(['-400', 'targ', '400'])
        if fig_4D:
            # movement ticks
            bax1.axs[1].set_xticks([-300, 0])
            bax1.axs[1].set_xticklabels(['-300', 'move'])
        else:
            # movement ticks
            bax1.axs[1].set_xticks([1250, 1550, 2170])
            bax1.axs[1].set_xticklabels(['-300', 'move', '600'])
    else:
        # preparatory ticks
        bax1.axs[0].set_xticks([0, 400, 800])
        bax1.axs[0].set_xticklabels(['-400', 'targ', '400'])
        if fig_4D:
            # movement ticks
            bax1.axs[1].set_xticks([-300, 0])
            bax1.axs[1].set_xticklabels(['-300', 'move'])
        else:
            # movement ticks
            bax1.axs[1].set_xticks([1170, 1470, 2090])
            bax1.axs[1].set_xticklabels(['-300', 'move', '600'])

    # sets titles and legend  
    bax1.set_ylabel("Variance")
    if PMd:
        an_text = "PMd to M1"
    else:
        an_text = "Neurons to Muscles"
    title_text = f"Monkey {J_text} {an_text} Variance"
    if fig_4D:
        title_text = f"Monkey {J_text} {an_text} Tuning"
        bax1.set_ylabel("tuning")
    bax1.set_title(title_text)
    bax1.set_xlabel("Time")
   
    bax1.legend(loc = 2)
