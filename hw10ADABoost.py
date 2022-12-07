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
from sklearn.ensemble import AdaBoostClassifier


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

    feature_mode='haar'
    max_size=1
    X_train, Y_train = prep_data_AdaBoost(train_pos_img_path, train_neg_img_path, feature_mode, max_size)
    X_test, Y_test = prep_data_AdaBoost(test_pos_img_path, test_neg_img_path,feature_mode, max_size)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    cascade_stages=[]
    s=1
    stop_condition=[1e-6,1e-6]
    MODEL_STAT = np.array([1,1])
    while 1:
        print("Stage: ",s)
        X_train, Y_train, model_stat, cascade = stage(X_train, Y_train)
        cascade_stages.append(cascade)
        MODEL_STAT = np.append(MODEL_STAT, model_stat)
        cond_vals = rearrange(MODEL_STAT,'(c h)->c h',h=2)
        cond_vals = np.cumprod(cond_vals, axis=0)
        s+=1
        if (cond_vals[-1]<stop_condition).all(): break

    print(cond_vals.shape)
    cas_vals = (np.arange(len(cond_vals))).astype(np.uint8)
    plt.plot(cas_vals, cond_vals[:,0], label='FP')
    plt.plot(cas_vals, cond_vals[:,1], label='FN')
    plt.ylabel('FP_FN')
    plt.xlabel('Num stages')
    plt.title('Plot of FP and FN VS stages')
    plt.legend()
    plt.show()


    test_AdaBoost(X_test, Y_test, cascade_stages)

    # test_num_pos = np.sum(Y_test==1)
    # for cascade in cascade_stages:
    #     cls = test_stage(cascade, X_test)
    #     model_stat = find_ft_pn(cls, test_num_pos)
    #     print(model_stat)
    #     X_test = X_test[np.where(cls==1), :][0]
    #     Y_test = Y_test[np.where(cls==1)[0]]
    #     test_num_pos = np.sum(Y_test==1)






if __name__ == '__main__':
    main()