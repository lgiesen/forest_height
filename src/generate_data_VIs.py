from os import listdir
from os.path import isfile, join
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

root_path = 'drive/MyDrive/Colab Notebooks/data/'
path_images = f'{root_path}images/'
path_masks = f'{root_path}masks/'


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

def extract_data(path_images, path_masks):
    """
    Extract data from zipped files

    Parameters
    ----------
    path_images: String
    path_masks: String
    Path to the train data (default: None)

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
    df: pandas.DataFrame
    """
    # extract non-zero value indices from y (= label position) to extract the corresponding X-value
    # prepare data to merge it into one data frame, 
    # which makes it easier to extract the values of the same pixel
    X = X.reshape(10, -1)
    y = y.reshape(1, -1)
    Xy = np.concatenate((X, y), axis=0)
    Xy = Xy.transpose()
    data = np.empty((0,11))
    data = np.concatenate((data, Xy), axis=0)
    
    indices = np.nonzero(data[:,-1])
    labeled_data = data[indices]

    # create dataframe with features and labels
    df = pd.DataFrame(labeled_data)
    df.columns = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'Label']
    
    return df

def upsample_data(df):
    """
    Upsample underrepresented data

    Parameters
    ----------
    df: pandas.DataFrame

    Returns
    -------
    features: pandas.DataFrame
    labels: pandas.DataFrame
    """

    # sort data according to tree height asc
    dfs = df.sort_values('Label').reset_index(drop=True) 
    # create empty data frame to fill
    dff = pd.DataFrame(columns=df.columns) 

    index_start = 0
    for i in range(3, 37, 3):
        #count the number of intances that are in one interval for example 0 - 3 or 15 - 18
        index_end = index_start + dfs["Label"][(dfs["Label"] > i - 3) & (dfs["Label"] < i)].count() 
        # take random smaple of the interval
        samp = dfs[index_start:index_end].sample(800) 
        dff = pd.concat((dff, samp))
        index_start = index_end

    # add the highest values beacuase there are only a few
    dff = pd.concat((dff, dfs[index_start:]))
    dftr = dff.sample(frac=1).reset_index(drop=True) #shuffel the dataset randomly
    
    # extract features and labels
    features = dftr.iloc[:, 0:10] 
    labels = dftr.iloc[:,10]
    
    # the length of X and y has to be the same
    # assert features.shape[0] == labels.shape[0]
    return (features, labels)

def calculate_ndvi(X):
    """
    Generate a dataset (X_train, X_test, y_train, y_test) based on the location of zip files

    Parameters
    ----------
    X: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # Extract the relevant bands for NDVI calculation
    b4, b8 = X['B4'], X['B8']
    # Calculate NDVI
    ndvi = (b8 - b4) / (b8 + b4)

    # Add NDVI as a new feature to X
    X["NDVI"] = ndvi
    return X

def calculate_VIs(X):
    """
    Generate a dataset (X_train, X_test, y_train, y_test) based on the location of zip files

    Parameters
    ----------
    X: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # Extract the relevant bands for calculating VIs
    b2, b4, b5, b6 = X['B2'], X['B4'], X['B5'], X['B6']
    b7, b8 = X['B7'], X['B8']

    # Enhanced Vegetation Index
    evi = 2.5 * ((b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1)) 

    # Soil Adjusted Vegetation Index (SAVI) 
    savi = (b8 - b4) / (b8 + b4 + 0.428) * (1.428)

    # Inverted Red-Edge Chlorophyll Index
    ireci =  b7 - b4 / (b5 / b6) 

    # Sentinel 2 Red Edge Position
    s2rep = 705 + 35 * ((b7 + b4) / 2 - b5) / (b6 - b5) 

    # Add VIs as a new feature to X
    X["EVI"] = evi
    X["SAVI"] = savi 
    X["IRECI"] = ireci
    X["s2rep"] = s2rep
    return X


def generate_dataset(path_images, path_masks, output_variables):
    """
    Generate a dataset (X_train, X_test, y_train, y_test) based on the location of zip files

    Parameters
    ----------
    path_images: String
    path_masks: String
    Path to the train data (default: None)
    output_variables: List

    Returns
    -------
    pd.DataFrame
    """
    X, y = extract_data(path_images, path_masks)
    df = extract_labels(X, y)
    del X, y
    features, labels = upsample_data(df)
    del df

    # check for each output variable
    if !'color_channels' in output_variables:
        features = pd.DataFrame()
    if 'NDVI' in output_variables:
        features = calculate_ndvi(features)
    if 'VI' in output_variables: 
        features = calculate_VIs(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0, shuffle=True)
    del features, labels
    return (X_train, X_test, y_train, y_test)