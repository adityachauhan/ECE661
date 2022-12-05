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



if __name__ == '__main__':
    main()