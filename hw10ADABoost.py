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
from sklearn.neighbors import KNeighborsClassifier as knn
from einops import rearrange
config = configparser.ConfigParser()
config.read('hw10config.txt')


def main():
    top_dir = config['PARAMETERS']['top_dir']
    data_dir = config['PARAMETERS']['data_dir']
    train_dir = config['PARAMETERS']['train_dir']
    test_dir = config['PARAMETERS']['test_dir']
    pos_dir = config['PARAMETERS']['positive']
    neg_dir = config['PARAMETERS']['negative']
    training_data_path = os.path.join(top_dir, data_dir, train_dir)
    testing_data_path = os.path.join(top_dir, data_dir, test_dir)
    # print(training_data_path)
    train_pos = os.path.join(top_dir, data_dir, train_dir, pos_dir)
    train_neg = os.path.join(top_dir, data_dir, train_dir, neg_dir)
    test_pos = os.path.join(top_dir, data_dir, test_dir, pos_dir)
    test_neg = os.path.join(top_dir, data_dir, test_dir, neg_dir)
    train_pos_img_path = glob.glob(train_pos+'/*.png')
    train_neg_img_path = glob.glob(train_neg+'/*.png')
    test_pos_img_path = glob.glob(test_pos + '/*.png')
    test_neg_img_path = glob.glob(test_neg + '/*.png')
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    num_pos_samples = len(train_pos_img_path)
    num_neg_samples = len(train_neg_img_path)
    init_weights_pos = np.ones((num_pos_samples))*(1/(2*num_pos_samples))
    init_weights_neg = np.ones((num_neg_samples))*(1/(2*num_neg_samples))
    init_weights = np.hstack((init_weights_pos,init_weights_neg))
    for i in range(len(train_pos_img_path)):
        path = train_pos_img_path[i]
        img = readImgCV(path)
        gray_img = bgr2gray(img)
        feature = get_feature(gray_img)
        X_train.append(feature.tolist())
        Y_train.append(1)

    for i in range(len(train_neg_img_path)):
        path = train_neg_img_path[i]
        img = readImgCV(path)
        gray_img = bgr2gray(img)
        feature = get_feature(gray_img)
        X_train.append(feature.tolist())

        Y_train.append(0)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    print(X_train.shape)
    print(Y_train.shape)

    # best_cls = classifier(X_train, Y_train, init_weights)
    # print(best_cls)
    cascade_stages=[]
    cum_fp=1
    cum_fn=1
    fp_vec=[]
    fn_vec=[]
    for i in range(10):
        features, labels, perfRates, cascade = stage(X_train, Y_train,10, init_weights, num_pos_samples, num_neg_samples)
        cascade_stages.append(cascade)
        fp = perfRates[0]
        fn = perfRates[1]
        cum_fp = cum_fp*fp
        cum_fn = cum_fn*fn
        fp_vec.append(cum_fp)
        fn_vec.append(cum_fn)
        if cum_fp<1e-6 and cum_fn<1e-6:
            break
        if np.sum(labels==0)==0:
            break

    cas_vals = (np.arange(len(fp_vec))+1).astype(np.uint8)
    plt.plot(cas_vals, fp_vec)
    plt.plot(cas_vals, fn_vec)
    plt.ylabel('cum rate')
    plt.xlabel('num_cas')
    plt.title('plot')
    plt.show()






if __name__ == '__main__':
    main()