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



base_path =r"c:\Users\katie\OneDrive\Desktop\Thesis"
with open(base_path+'\J_neu.pkl', "rb") as input_file:
    J_pickle = pickle.load(input_file)
del input_file
#help


base_path =r"c:\Users\katie\OneDrive\Desktop\Thesis"
with open(base_path+'\J_mus.pkl', "rb") as input_file:
    J_pickle_m = pickle.load(input_file)
del input_file

# base_path = "/Users/kb6113/Desktop/Thesis"
# with open(base_path+'/J_neu.pkl', "rb") as input_file:
#     J_pickle = pickle.load(input_file)
# del input_file

# with open(base_path+'/J_mus.pkl', "rb") as input_file:
#     J_pickle_m = pickle.load(input_file)
# del input_file

J_all_PSTH = J_pickle['J_all']['interpPSTH']
J_all_PSTH_new = pcar.scaling(J_all_PSTH)
print(J_all_PSTH_new.shape)
# pcar.frac_var(J_all_PSTH_new, .7, plot=True)

# testing if lambda and solving for W works
print(J_all_PSTH_new[0:202, 0])
print(np.average(np.max(J_all_PSTH_new, axis = 0)))
print(np.average(np.min(J_all_PSTH_new, axis = 0)))


W_T = np.random.uniform(1, 9, 18).reshape((6, 3))
low_N = np.random.uniform(0.2, 20.59, 120)
low_N = np.reshape(low_N,(20,6))
low_M = low_N @ W_T
print (low_M.shape)

print (low_N.shape)
print (W_T.shape)
print (low_M.shape)

W_hat, M_hat, _ = pcar.regress(low_M, low_N, lam = 0.001)
MSE = np.mean((low_M - M_hat)**2)
print(W_T)
print(W_hat)
print(MSE)
