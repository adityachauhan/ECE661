import random
import numpy as np
from vision import *
import configparser
from tqdm import tqdm
from utility import *
import matplotlib.pyplot as plt
import os
import cv2
import glob
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
from einops import rearrange
config = configparser.ConfigParser()
config.read('hw10config.txt')


def main():
    top_dir = config['PARAMETERS']['top_dir']
    data_dir = config['PARAMETERS']['data_dir']
    train_dir = config['PARAMETERS']['train_dir']
    test_dir = config['PARAMETERS']['test_dir']
    training_data_path = os.path.join(top_dir, data_dir, train_dir)
    testing_data_path = os.path.join(top_dir, data_dir, test_dir)
    X_train, Y_train = data_loader(training_data_path)
    X_test, Y_test = data_loader(testing_data_path)
    X_test, m_test = prep_data(X_test)
    X_train_prep, m_train_prep = prep_data(X_train)
    num_classes=30
    num_samples=21
    X_mean, W = PCA(X_train)
    print(W.shape)
    idx_array = idx_per_class(Y_train, num_classes, num_samples)
    print(idx_array.shape)
    W_lda = LDA(X_train, idx_array)
    print(W_lda.shape)
    num_test_sample = len(X_test)
    print(num_test_sample)
    K = np.arange(10)
    for k in K:
        y_test_pca = W[:k,:]@X_test.T
        y_train_pca = W[:k,:]@X_train_prep.T
        y_test_lda = W_lda[:k, :] @ X_test.T
        y_train_lda = W_lda[:k, :] @ X_train_prep.T
        # print(y_test.shape, y_train.shape)
        num_matches_pca = nn(y_test_pca.T,y_train_pca.T,Y_train)
        num_matches_lda = nn(y_test_lda.T,y_train_lda.T,Y_train)
        print(num_matches_pca, num_matches_lda)
    # print(W.shape, X_test.shape)


if __name__ == '__main__':
    main()