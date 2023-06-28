from os import listdir
from os.path import isfile, join
from zipfile import ZipFile

import numpy as np

root_path = 'drive/MyDrive/Colab Notebooks/data/'
path_images = f'{root_path}images/'
path_masks = f'{root_path}masks/'

def get_files(dir):
    return [f for f in listdir(dir) if isfile(join(dir, f))]

def extract_data(data_filenames):
    """
    Extract data from zipped files

    Parameters
    ----------
    data_filenames: Array 
    Path to the train data (default: None)
    root_path + filename = complete filepath 

    Returns
    -------
    dataset: Tuple of np.ndarray
    """
    # unzip data
    for filename in data_filenames:
        with ZipFile(filename, 'r') as zObject:
            zObject.extractall(path=f"{root_path}{filename.replace('.zip','')}")

    # load satellite images
    X = np.load(f'/content/images/{get_files(path_images)[0]}')
    for filename in get_files(path_images)[1:]:
        temp = np.load(f'/content/images/{filename}', allow_pickle=True)
        X = np.concatenate((X, temp))
    # load labels
    y = np.load(f'/content/masks/{get_files(path_masks)[0]}')
    for filename in get_files(path_masks)[1:]:
        temp = np.load(f'/content/masks/{filename}', allow_pickle=True)
        y = np.concatenate((y, temp))
    del temp

    # ceil the values at 2000 because clouds have a different reflection value
    ceiling = 2000
    X[X > ceiling] = ceiling
    #scale values between 0 and 1
    X = X / ceiling
    
    return (X, y)