import numpy as np 
def compute_MSLL(ytrue, ypred, ypred_var, train_mean=None, train_var=None):
    """ 
    Computes the MSLL or MLL (not standardized) if 'train_mean' and 'train_var' are None.

        Basic usage::

            MSLL = compute_MSLL(ytrue, ypred, ytrue_sig, noise_variance, train_mean, train_var)

        where

        :param ytrue     : n*p matrix of true values where n is the number of samples 
                           and p is the number of features. 
        :param ypred     : n*p matrix of predicted values where n is the number of samples 
                           and p is the number of features. 
        :param ypred_var : n*p matrix of summed noise variances and prediction variances where n is the number of samples 
                           and p is the number of features.

        :param train_mean: p dimensional vector of mean values of the training data for each feature.

        :param train_var : p dimensional vector of covariances of the training data for each feature.

        :returns loss    : p dimensional vector of MSLL or MLL for each feature.

    """

    if train_mean is not None and train_var is not None:

        # make sure y_train_mean and y_train_sig have right dimensions (subjects x voxels):
        Y_train_mean = np.repeat(train_mean, ytrue.shape[0], axis=0)
        Y_train_sig = np.repeat(train_var, ytrue.shape[0], axis=0)

        # compute MSLL:
        loss = np.mean(0.5 * np.log(2 * np.pi * ypred_var) + (ytrue - ypred)**2 / (2 * ypred_var) -
                       0.5 * np.log(2 * np.pi * Y_train_sig) - (ytrue - Y_train_mean)**2 / (2 * Y_train_sig), axis=0)

    else:
        # compute MLL:
        loss = np.mean(0.5 * np.log(2 * np.pi * ypred_var) +
                       (ytrue - ypred)**2 / (2 * ypred_var), axis=0)

    return loss

def explained_var(ytrue, ypred):
    """ 
    Computes the explained variance of predicted values.

        Basic usage::

            exp_var = explained_var(ytrue, ypred)

        where

        :ytrue: n*p matrix of true values where n is the number of samples 
                and p is the number of features. 
        :ypred: n*p matrix of predicted values where n is the number of samples 
                and p is the number of features. 

        :returns exp_var: p dimentional vector of explained variances for each feature.

     """

    exp_var = 1 - (ytrue - ypred).var(axis=0) / ytrue.var(axis=0)

    return exp_var