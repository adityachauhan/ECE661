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
    pos_dir = config['PARAMETERS']['positive']
    neg_dir = config['PARAMETERS']['negative']

    train_pos = os.path.join(top_dir, data_dir, train_dir, pos_dir)
    train_neg = os.path.join(top_dir, data_dir, train_dir, neg_dir)
    test_pos = os.path.join(top_dir, data_dir, test_dir, pos_dir)
    test_neg = os.path.join(top_dir, data_dir, test_dir, neg_dir)

    train_pos_img_path = glob.glob(train_pos+'/*.png')
    train_neg_img_path = glob.glob(train_neg+'/*.png')
    test_pos_img_path = glob.glob(test_pos + '/*.png')
    test_neg_img_path = glob.glob(test_neg + '/*.png')

    feature_mode='sobel'
    max_size=1
    X_train, Y_train = prep_data_AdaBoost(train_pos_img_path, train_neg_img_path, feature_mode, max_size)
    X_test, Y_test = prep_data_AdaBoost(test_pos_img_path, test_neg_img_path,feature_mode, max_size)


    cond_vals, cascade_stages = train_AdaBoost(X_train, Y_train)
    print(cond_vals.shape)
    plot_AdaBoost_train_curve(cond_vals)


    model_stat = test_AdaBoost(X_test, Y_test, cascade_stages)
    ada_boost_acc = AdaBoost_accuracy(model_stat, Y_test.shape[0])
    print("AdaBoost model accuracy: "+str(ada_boost_acc)+"%")






if __name__ == '__main__':
    main()