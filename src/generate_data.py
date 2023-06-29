from os import listdir
from os.path import isfile, join
from zipfile import ZipFile

import numpy as np

root_path = 'drive/MyDrive/Colab Notebooks/data/'
path_images = f'{root_path}images/'
path_masks = f'{root_path}masks/'

def get_files(dir):
    """
    Get all files from a directory

    Parameters
    ----------
    dir: Array 

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

    # load labels by loading the first one and then concatenating the rest
    y = np.load(f'{path_masks}{get_files(path_masks)[0]}')
    for filename in get_files(path_masks)[1:]:
        temp = np.load(f'{path_masks}{filename}', allow_pickle=True)
        y = np.concatenate((y, temp))
  
    # ceil the values at 2000 because clouds have a different reflection value
    ceiling = 2000
    X[X > ceiling] = ceiling
    #scale values between 0 and 1
    X = X / ceiling
    del temp, ceiling
    
    return (X, y)