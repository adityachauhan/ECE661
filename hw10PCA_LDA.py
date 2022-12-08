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
config = configparser.ConfigParser()
config.read('hw10config.txt')


def main():
    top_dir = config['PARAMETERS']['top_dir']
    data_dir = config['PARAMETERS']['data_dir']
    train_dir = config['PARAMETERS']['train_dir']
    test_dir = config['PARAMETERS']['test_dir']
    training_data_path = os.path.join(top_dir, data_dir, train_dir)
    testing_data_path = os.path.join(top_dir, data_dir, test_dir)
    X_train, Y_train,mg_train = data_loader(training_data_path)
    X_test, Y_test,mg_test = data_loader(testing_data_path)
    num_classes=30
    W_pca = PCA(X_train, mg_train)
    idx_array = idx_per_class(Y_train, num_classes)
    W_lda = LDA(X_train,mg_train,idx_array, num_classes)
    K = np.arange(29)
    pca_acc = []
    lda_acc = []
    tot_samples=len(X_test)
    for k in K:
        y_test_pca = W_pca[:,:k+1].T@(X_test-mg_test)
        y_train_pca = W_pca[:,:k+1].T@(X_train-mg_train)
        y_test_lda = W_lda[:, :k+1].T @ (X_test-mg_test)
        y_train_lda = W_lda[:, :k+1].T @(X_train-mg_train)
        num_matches_pca = NearestNeighbor(y_test_pca.T,y_train_pca.T,Y_train,Y_test)
        num_matches_lda = NearestNeighbor(y_test_lda.T,y_train_lda.T,Y_train,Y_test)
        print(k, num_matches_pca, num_matches_lda)
        pca_acc.append(num_matches_pca/tot_samples)
        lda_acc.append(num_matches_lda/tot_samples)
    pca_acc = np.array(pca_acc)
    lda_acc = np.array(lda_acc)
    plot_acc(pca_acc, lda_acc, K)

if __name__ == '__main__':
    main()