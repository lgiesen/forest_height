from os import listdir
from os.path import isfile, join
from zipfile import ZipFile

import numpy as np


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
    os.system('mkdir images')
    os.system('mkdir masks')
    # unzip data
    for idx, filename in enumerate(data_filenames):
        if filename[0][-4:] != '.zip':
            print(f'{filename} cannot be extracted because it is not a zip file.')
            continue;
        data_filenames[idx] = filename.replace('.zip','')
        mask_or_img = 'masks' if 'mask' in filename else 'images'
        with ZipFile(filename, 'r') as zObject:
            zObject.extractall(path=f"{root_path}{mask_or_img}/{filename}")
    
    # load satellite images by loading the first one and then concatenating the rest
    X = np.load(f'{path_images}{get_files(path_images)[0]}')
    for filename in get_files(path_images)[1:]:
        temp = np.load(f'{path_images}{filename}', allow_pickle=True)
        X = np.concatenate((X, temp))
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
    del mask_or_img, temp, ceiling
    
    return (X, y)