from os import listdir
from os.path import isfile, join

import numpy as np
from google.colab import drive

root_path = 'drive/MyDrive/Colab Notebooks/data/'
path_images = f'{root_path}images/'
path_masks = f'{root_path}masks/'
drive.mount ('/content/drive', force_remount=True)

def get_files(dir):
    return [f for f in listdir(dir) if isfile(join(dir, f))]

def extract_data(data_path):
    """
    Extract data from zipped files

    Parameters
    ----------
    data_path: Array 
    Path to the train data (default: None)

    Returns
    -------
    dataset: tf.data.Dataset
    """
    # unzip data
    # for path in data_path:
    #     !unzip path

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

extract_data([f'{path_images}images_train.zip', f'{path_masks}masks_train.zip'])

# remove drive connection as it is no longer needed
drive.flush_and_unmount()