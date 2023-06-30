from os import listdir
from os.path import isfile, join
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import train_test_split


def get_files(dir):
    """
    Get all files from a directory

    Parameters
    ----------
    dir: Array of strings 

    Returns
    -------
    Array of strings
    """
    return [f for f in listdir(dir) if isfile(join(dir, f))]

def extract_data(data_filenames):
    """
    Extract data from zipped files

    Parameters
    ----------
    data_filenames: Array of strings
    Path to the train data (default: None)
    root_path + filename = complete filepath 

    Returns
    -------
    dataset: Tuple of np.ndarray
    """

    # load satellite images by loading the first one and then concatenating the rest
    X = np.load(f'{path_images}{get_files(path_images)[0]}')
    for filename in get_files(path_images)[1:]:
        temp = np.load(f'{path_images}{filename}', allow_pickle=True)
        X = np.concatenate((X, temp))
    # reshape X to distinguish between image and color channel
    num_imgs = len(get_files(path_images))
    X = X.reshape((num_imgs, int(X.shape[0]/num_imgs), X.shape[1], X.shape[2]))
    # ceil the values at 2000 because clouds have a different reflection value
    ceiling = 2000
    X[X > ceiling] = ceiling
    #scale values between 0 and 1
    X = X / ceiling

    # load labels by loading the first one and then concatenating the rest
    y = np.load(f'{path_masks}{get_files(path_masks)[0]}')
    for filename in get_files(path_masks)[1:]:
        temp = np.load(f'{path_masks}{filename}', allow_pickle=True)
        y = np.concatenate((y, temp))
  
    del temp, ceiling, num_imgs
    
    return (X, y)

def extract_labels(X, y):
    """
    Labels are sparse, so they are Get all labels (non-zero elements) from a set of images

    Parameters
    ----------
    X: numpy.ndarray
    y: numpy.ndarray

    Returns
    -------
    Tuple of numpy.ndarray
    """
    # extract non-zero value indices from y (= label position) to extract the corresponding X-value
    # this has to be done for every image, because X has a different length than y, 
    # if both are flattened due to more color channels
    X_labeled = X[0].flat[np.nonzero(y[0].flat)[0]]
    
    for img_idx in range(1, y.shape[0]):
        cur_img_nonzero_indices = np.nonzero(y[img_idx].flat)[0]
        corresponding_cur_X_values = X[img_idx].flat[cur_img_nonzero_indices]
        X_labeled = np.concatenate((X_labeled, corresponding_cur_X_values))

    # y just has one dimension, so no loop is needed
    y_labeled = y.flat[np.nonzero(y.flat)[0]]
    # the length of X and y has to be the same
    assert y_labeled.shape == X_labeled.shape
    del corresponding_cur_X_values, cur_img_nonzero_indices
    return (X_labeled, y_labeled)

def generate_dataset(zip_files):
    """
    Generate a dataset (X_train, X_test, y_train, y_test) based on the location of zip files

    Parameters
    ----------
    zip_files: Array of strings

    Returns
    -------
    Numpy.ndarray
    """
    X, y = extract_data(zip_files)
    del zip_files
    X_labeled, y_labeled = extract_labels(X, y)
    del X, y
    X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=0, shuffle=True)
    del X_labeled, y_labeled
    return (X_train, X_test, y_train, y_test)