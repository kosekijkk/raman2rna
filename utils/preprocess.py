
'''
DESCRIPTION:
Script containing useful functions for pre-processing Raman data. Generally, the following functions are the
most useful and should be used in the following order:
1. get_fingerprint_region
2. remove_sharp_peaks_2d_iter
3. remove_fluorescence
4. subtract_mean_horizontal

Note that Standard Scaling is not included, but can easily be implemented using sklearn's
StandardScaler().fit_transform(matrix) function
'''

# native imports
import numpy as np
import pandas as pd

# third part imports
import rampy as rp

# local imports



def to_dataframe(matrix, label_df, column_names=None):
    '''
        Given a matrix (numpy), converts to pandas DataFrame with given
        labels and column names. Labels are needed for easy plotting by day

        @:param
            - matrix - numpy matrix of spectra
            - label_df - Pandas DataFrame of labels of first dimension as matrix
            - column names (optional) - column names of the spectra. Useful if you want to
                replace a list of integers with a list of strings or vice-versa, etc.

        @:returns
            - Pandas DataFrame of the labels concatenated to the spectra
    '''

    assert matrix.shape[0] == label_df.shape[0], 'matrix and labels are not the same shape, cannot be concatenated'
    if column_names:
        # cast to pandas matrix with column names
        spectra_df = pd.DataFrame(data=matrix, columns=column_names)
    else:
        spectra_df = pd.DataFrame(data=matrix)
    # concatenate labels with matrix along axis=1
    return pd.concat([label_df, spectra_df], axis=1) 


def get_fingerprint_region(matrix, start_ix=410, end_ix=None):
    '''
        Slices matrix to extract fingerprint region defined by df[:,start_ix:end_ix]

        @:param
            - matrix - numpy matrix of spectra
            - start_ix - the index of the first column in the slice. Default is 410
            - end_ix - the index of the last column in the slice. Default is None, meaning everything
                       including and after start_ix will be in the slice

        @:returns
            - Numpy matrix, representing the slice of a spectra
    '''
    assert isinstance(matrix, np.ndarray), 'input matrix is of type {}, but should be of type np.ndarray'.format(type(matrix))
    
    if end_ix: return matrix[:, start_ix:end_ix]
    else: return matrix[:, start_ix:]
    
    # TODO - maybe add pandas dataframe functionality later
#     elif isinstance(df, pd.core.frame.DataFrame):
#         if end_ix: return pd.concat([df.loc[:, :last_col_label], df.loc[:,start_ix:end_ix]], axis=1)
#         else: return pd.concat([df.loc[:, :last_col_label], df.loc[:,start_ix:]], axis=1)
#     return -1

#####################################################
################### Peak Removal ####################
#####################################################

def _get_peaks(intensity, threshold):
    '''
    Helper function for remove_sharp_peaks_1d_iter: Produces a mask where True values have a peak
    greater than given threshold

        @:param:
        - intensity is a 1-D Raman intensity vector
        - threshold is the z-score threshold, above which an intensity is considered a sharp peak


        @:returns
        - numpy array of the peak mask, where True values mean there's a peak at a given index and False
            means no peak exists at that index
    '''

    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = np.abs(0.6745 * (intensity - median_int) / mad_int)
    peaks = modified_z_scores > threshold
    return peaks

def remove_sharp_peaks_1d_iter(intensity, win_size=10, threshold=5, max_iter=10):
    '''
        Removes peaks from intensity vector by replacing them with 
        the mean of non-peak values within a given window size
        
        Threshold represents the z-score threshold. We found that a threshold
        of 5 works well for the cellular reprogramming Raman dataset

        @:param:
        - intensity is a 1-D Raman intensity vector
        - win_size (int) - the window surrounding a peak considered for imputing the peak. Imputing is
                           done by taking the mean spectra of surrounding non-peak spectra within this
                           window. Default is 10
        - threshold (int) is the z-score threshold, above which an intensity is considered a sharp peak.
                          Default is 5
        - max_iter (int) is the maximum number of iterations used to remove a peak. There are some
                         abnormal peaks that can only be corrected with multiple application of
                         the Whitaker-Hayes method:

        @:returns
        - numpy array of the Raman intensity, now with peaks removed based on the above parameters

        References:
        https://chemrxiv.org/articles/A_Simple_Algorithm_for_Despiking_Raman_Spectra/5993011/2?file=10761493
        https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22

    '''
    n = len(intensity)
    delta_int = np.diff(intensity) #take first derivative to show sharp jumps
    peaks = _get_peaks(delta_int, threshold)
    num_peaks = sum(peaks)
    y_out = intensity.copy()
    win_ind = np.arange(n)
    iter_ = 0
    while iter_ < max_iter and num_peaks > 0: #include while loop to iteratively remove peaks
        for i in np.arange(len(peaks)):
            # if we have a peak, replace with mean of surrounding non-peaks
            if peaks[i] > 0:
                # get surrounding window indices
                # prevent out of boundary errors
                if i-win_size <0: lower = 0
                else: lower = i-win_size

                if i+1+win_size >= n: upper = n-1
                else: upper = i+1+win_size

                win_ind = np.arange(lower, upper)
    #             print(f'window: {win_ind}')
                win_mask = win_ind[peaks[win_ind] == 0] # take the ones that are not peaks
                try:
                    y_out[i] = np.mean(intensity[win_mask]) # take the mean of these surrounding values
                except: #TODO - changing this to be a general except instead of except ValueError
                    y_out[i] = intensity[i] # if NaN, just replace with original peak
        # get new number of peaks after correction
        delta_int = np.diff(y_out) #take first derivative to show sharp jumps
        peaks = _get_peaks(delta_int, threshold)
        num_peaks = sum(peaks)
        iter_ += 1
    return y_out


def remove_outliers(intensity, win_size=7, threshold=5):
    y_out = intensity.copy()
    y_out_med = sp.signal.medfilt(intensity, win_size)
    y_outliers = (y_out-y_out_med)>threshold
    y_out[y_outliers] = y_out_med[y_outliers]

    return y_out



def remove_sharp_peaks_2d_iter(matrix, win_size=10, threshold=5, axis=1, max_iter=20):
    '''
        Applies remove_sharp_peaks_2d_iter to a 2D matrix:

        Removes peaks from a matrix of intensity vectors by replacing them with
        the mean of non-peak values within a given window size

        Threshold represents the z-score threshold. We found that a threshold
        of 5 works well for the cellular reprogramming Raman dataset

        @:param:
        - intensity is a 1-D Raman intensity vector
        - win_size (int) - the window surrounding a peak considered for imputing the peak. Imputing is
                           done by taking the mean spectra of surrounding non-peak spectra within this
                           window. Default is 10
        - threshold (int) is the z-score threshold, above which an intensity is considered a sharp peak.
                          Default is 5
        - max_iter (int) is the maximum number of iterations used to remove a peak. There are some
                         abnormal peaks that can only be corrected with multiple application of
                         the Whitaker-Hayes method:

        @:returns
        - numpy array of the Raman intensity, now with peaks removed based on the above parameters

        References:
        https://chemrxiv.org/articles/A_Simple_Algorithm_for_Despiking_Raman_Spectra/5993011/2?file=10761493
        https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22

    '''
    return np.apply_along_axis(remove_sharp_peaks_1d_iter, 
                        axis=axis, 
                        arr=matrix, 
                        win_size=win_size, 
                        threshold=threshold,
                        max_iter=max_iter)


#####################################################
########## Autofluorescence correction ##############
#####################################################


def remove_fluorescence(matrix, method='als', x_axis=np.arange(410,1340)):
    ''' 
        Applies flouorescence removal using rampy baseline correction
        to each row of a matrix
        @:param:
        - matrix is numpy array of spectra before fluorescence correction
        - method is one of ['poly', 'als', 'arPLS', 'drPLS']. Default is 'als' or alternating least-squares
        - x_axis represents the tick marks of the x_asis. Default is np.arange(410, 1340) or the
                fingerprint region.  Change if you're using a different fingerprint region

        @:returns:
        - np array of spectra after baseline fluorescence correction (spectra_bc)
    '''
    assert len(x_axis) == matrix.shape[1], 'x_axis ({}) and matrix ({}) are not the same dimension'.format(x_axis.shape, matrix.shape)
    # find parameters for rampy.baseline
    x_axis[0], x_axis[-1]
    lower_y, upper_y = np.min(matrix), np.max(matrix)

    # apply a wrapper to rampy.baseline function using lambda function
    baseline_wrapper = lambda y, x, bir, method: rp.baseline(x, y, bir, method)

    # apply function along axis
    spectra_bc_tup = np.apply_along_axis(baseline_wrapper, axis=1, arr=matrix, x=x_axis, method=method,
                                         bir=np.array([[x_axis[0], x_axis[-1]],[lower_y, upper_y]]))

    # remove spectra and extra dimensions
    spectra_bc = spectra_bc_tup[:,0,:,:].squeeze()
    return spectra_bc


#####################################################
############## Mean substraction ####################
#####################################################

def subtract_mean_horizontal(matrix):
    '''
        standardizes by subtracting mean horizontally, moving centering
        all spectra around zero

        @:param:
        - matrix is a 2D numpy array

        @:returns
        - 2D numpy array, now centered around 0
    '''
    return matrix - np.mean(matrix, axis=1, keepdims=True)



#####################################################
##### Low pass (Savitzky Golay) filtering ###########
#####################################################

def low_pass_filter(matrix, polyorder=3, window_length=9):
    '''  Applies savgol filter to all spectra in a matrix'''
    return np.apply_along_axis(savgol_filter, 
                        axis=1, 
                        arr=matrix, 
                        window_length=window_length, 
                        polyorder=3)
