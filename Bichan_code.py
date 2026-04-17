import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def svd_RRR(X, Y, rnk, lambda_=0):
    """
        Perform Ridge Regularized Reduced Rank Regression (RRR) using SVD.
        Parameters:
            X : np.ndarray
                Input data matrix (n_samples, n_input_neurons).
            Y : np.ndarray
                Output data matrix (n_samples, n_output_neurons).
            rnk : int
                Dimensionaility of communication
            lambda_ : float
                Regularization parameter (default is 0 for no regularization).
        Returns:
            w0 : np.ndarray
                Estimate of the communication strength (n_input_neurons, n_output_neurons).
            urrr : np.ndarray
                Input axes (n_input_neurons, rnk).
            vrrr : np.ndarray
                Output axes, orthonormal (n_output_neurons, rnk).
    """
    # Check if X and Y are 2D arrays
    # Ridge regularization
    XX = X.T @ X + lambda_ * np.eye(X.shape[1])

    # Least squares estimate with ridge
    if np.linalg.cond(XX) < 1e10:
        wridge = np.linalg.solve(XX, X.T @ Y)
    else:
        wridge = np.linalg.pinv(XX) @ (X.T @ Y)

    # SVD of relevant matrix
    _, _, vrrr = np.linalg.svd(Y.T @ X @ wridge)

    # Get the top 'rnk' components
    vrrr = vrrr[:rnk, :].T   # shape: (features, rnk)
    urrr = wridge @ vrrr    # shape: (features, rnk)

    # Construct full RRR estimate
    w0 = urrr @ vrrr.T
    vrrr = vrrr.T  # for compatibility with original code's return

    return w0, urrr, vrrr


def svd_RRR_noniso(X, Y, rnk, C=None):
    """
        Perform Reduced Rank Regression (RRR) using SVD with non-isotropic noise.
        Parameters:
            X : np.ndarray
                Input data matrix (n_samples, n_input_neurons).
            Y : np.ndarray
                Output data matrix (n_samples, n_output_neurons).
            rnk : int
                Dimensionaility of communication
            C : np.ndarray, optional
                Covariance matrix of the noise (default is None, estimate from data).
        Returns:
            w0 : np.ndarray
                Estimate of the communication strength (n_input_neurons, n_output_neurons).
            urrr : np.ndarray
                Input axes (n_input_neurons, rnk).
            vrrr : np.ndarray
                Output axes (n_output_neurons, rnk).
    """
    # Least squares estimate
    wls = np.linalg.solve(X.T @ X, X.T @ Y)

    # Compute covariance of residuals if C is not provided
    if C is None:
        res_wls = Y - X @ wls
        C = (res_wls.T @ res_wls) / (X.shape[0] - 1)

    # Compute inverse sqrt and sqrt of C
    C_sqrt = sqrtm(C)
    C_inv_sqrt = inv(C_sqrt)

    # SVD of whitened cross-covariance
    _, _, vrrr = np.linalg.svd(C_inv_sqrt @ Y.T @ X @ wls @ C_inv_sqrt)
    vrrr = vrrr[:rnk, :].T  # shape: (features, rnk)

    # Adjust for non-isotropic noise
    vrrr = C_sqrt @ vrrr
    urrr = np.linalg.solve(X.T @ X, X.T @ Y @ inv(C) @ vrrr)

    # Reconstruct estimate
    w0 = urrr @ vrrr.T

    return w0, urrr, vrrr
    
def get_or(d, key, default):
    val = d.get(key, default)
    d[key] = val
    return val, d

def simu_RRR(ops=None):
    if ops is None:
        ops = {}

    # Load or default parameters
    T, ops = get_or(ops, 'T', 100)
    nx, ops = get_or(ops, 'nx', 10)
    ny, ops = get_or(ops, 'ny', 5)
    rnk, ops = get_or(ops, 'rnk', 1)
    signse, ops = get_or(ops, 'signse', 0.1)
    magnitude, ops = get_or(ops, 'magnitude', 1)
    thetas, ops = get_or(ops, 'thetas', [])
    Sigma, ops = get_or(ops, 'Sigma', [])
    U, ops = get_or(ops, 'U', [])
    V, ops = get_or(ops, 'V', [])

    # Generate input X
    X = np.random.randn(T, nx)
    X -= X.mean(axis=0)  # center each column

    # Generate communication matrix
    if U == [] or V == []:
        U = np.random.randn(nx, rnk)
        V = np.random.randn(ny, rnk)
        if len(thetas) > 0:
            U_svd, _, V_svd = np.linalg.svd(U @ V.T, full_matrices=False)
            U = U_svd[:, :rnk]
            V = V_svd[:rnk, :].T / np.sqrt(thetas)
        V = V * magnitude

    B = U @ V.T

    # Generate Y
    if len(Sigma) == 0:
        E = signse * np.random.randn(T, ny)
        # E -= E.mean(axis=0)
        Y = X @ B + E
    else:
        E = np.random.multivariate_normal(np.zeros(ny), Sigma, size=T)
        Y = X @ B + E

    return X, Y, U, V, ops


def alignment_output(X, Y, W):
    """
    Compute the alignment index of the output population activities with communication subspace.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_input_features)
        Input activities.
    Y : ndarray of shape (n_samples, n_output_features)
        Output activities.
    W : ndarray of shape (n_input_features, n_output_features)
        Communication weights.

    Returns
    -------
    outputalignmentidx : float
        Normalized alignment index (between -1 and 1).
    """

    # ----- calculate raw alignment index -----
    # Do PCA on population covariance
    upop, spopvec, _ = np.linalg.svd(np.cov(Y, rowvar=False))
    spopcum = np.cumsum(spopvec)
    spopvec_nrm = spopvec / np.sum(spopvec)
    spopcum_nrm = np.cumsum(spopvec_nrm)

    # Project communication covariance onto PCs
    cov_predicted = np.cov(X @ W, rowvar=False)
    scomvec = np.diag(upop.T @ cov_predicted @ upop)
    # scomvec_nrm = scomvec / np.sum(scomvec)
    scomcum_nrm = np.cumsum(scomvec) / np.sum(scomvec)

    # Compute alignment index
    muscom = np.mean(scomcum_nrm)
    muspop = np.mean(spopcum_nrm)
    alignment_raw = np.dot(scomvec, spopvec) # dot product of scomvec and spopvec

    # Compute communication fraction
    commfrac = np.sum(scomvec) / np.sum(spopvec)

    # ----- normalize alignment index -----
    totcom = np.sum(scomvec)

    # Max possible alignment
    ii = np.argmax(spopcum > totcom + 1e-10)
    scommax = spopvec.copy()
    scommax[ii:] = 0
    scommax[ii] = totcom - np.sum(scommax[:ii])
    scommax_cum = np.cumsum(scommax) / np.sum(scommax)
    a_max = np.dot(scommax, spopvec)

    # Min possible alignment (flip order of eigenvalues)
    spopvec_rev = np.flipud(spopvec)
    spopcum_rev = np.cumsum(spopvec_rev)
    ii = np.argmax(spopcum_rev > totcom + 1e-10)
    scommin = spopvec_rev.copy()
    scommin[ii:] = 0
    scommin[ii] = totcom - np.sum(scommin[:ii])
    scommin = np.flipud(scommin)  # flip back
    scommin_cum = np.cumsum(scommin) / np.sum(scommin)
    a_min = np.dot(scommin, spopvec)

    # Rescale alignment
    outputalignmentidx = (alignment_raw - a_min) / (a_max - a_min)

    return outputalignmentidx, commfrac


def alignment_input(X, W, r=None, C=None):
    """
    Calculate how much the communication weights W align with 
    the principal components of the input X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_input_neurons)
        Input matrix (stimuli).
    W : ndarray of shape (n_input_neurons, n_output_neurons)
        Communication weights.
    r : int, optional
        Rank of the communication weights W (if not provided,
        estimate from W).
    C : ndarray, optional
        Covariance matrix of X. If None, will compute cov(X).

    Returns
    -------
    aa : float
        Alignment index (0-1), where 1 is maximally aligned.
    p : None
        Placeholder (for compatibility with MATLAB signature).
    aa_rand : None
        Placeholder (for compatibility with MATLAB signature).
    """
    # Covariance of inputs
    if C is None:
        C = np.cov(X, rowvar=False)

    # PCA of covariance matrix
    _, Spcavec, _ = np.linalg.svd(C)

    # SVD of weights
    _, Swvec, _ = np.linalg.svd(W)
    # Pad singular values to length n_input_neurons
    Swvec_padded = np.concatenate([Swvec, np.zeros(W.shape[0] - len(Swvec))])

    # Compute alignment index
    amax = Spcavec @ (Swvec_padded ** 2)             # maximal value
    amin = Spcavec @ (np.flipud(Swvec_padded ** 2))  # minimal value
    araw = np.trace(W.T @ C @ W)                     # test statistic

    aa = (araw - amin) / (amax - amin)

    # Match MATLAB output signature
    return aa, None, None