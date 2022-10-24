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
from scipy.optimize import least_squares
from einops import rearrange

config = configparser.ConfigParser()
config.read('hw7config.txt')

def main():
    top_dir = config['PARAMETERS']['top_dir']
    data_dir = config['PARAMETERS']['data_dir']
    training_dir = config['PARAMETERS']['training_dir']
    training_dir = os.path.join(top_dir,data_dir,training_dir)
    testing_dir = config['PARAMETERS']['testing_dir']
    number_gram_samples = int(config['PARAMETERS']['number_gram_samples'])
    vgg_channels = int(config['PARAMETERS']['vgg_channels'])
    testing_dir = os.path.join(top_dir,data_dir,testing_dir)
    training_data = glob.glob(training_dir+'/*.jpg')
    testing_data = glob.glob(testing_dir+'/*.jpg')
    classes = config['PARAMETERS']['classes']
    classes = classes.split(',')
    labels_train=[]
    labels_test=[]
    texture_des_train_set=[]
    texture_des_test_set=[]
    model_path = os.path.join(top_dir, config['PARAMETERS']['vgg_model'])
    for i in trange(len(training_data)):
        img = readImgCV(training_data[i])
        if img is not None:
            name = training_data[i].split('.')[0].split('/')[-1]
            labels_train.append(getLabel(name))
            feature = vgg_feature_extractor(img, model_path)
            texture_des = channelNorm(feature)
            texture_des_train_set.append(texture_des)
        else:
            print("Cant open training file: ", training_data[i])

    for i in trange(len(testing_data)):
        img = readImgCV(testing_data[i])
        if img is not None:
            name = testing_data[i].split('.')[0].split('/')[-1]
            labels_test.append(getLabel(name))
            feature = vgg_feature_extractor(img, model_path)
            texture_des = channelNorm(feature)
            texture_des_test_set.append(texture_des)
        else:
            print("Cant open testing file: ", testing_data[i])
    output = config['PARAMETERS']['output']
    conf_plot_path = os.path.join(output, 'channelNormConfMat.png')
    classify(texture_des_train_set, labels_train, texture_des_test_set, labels_test, conf_plot_path, classes)




if __name__=='__main__':
    main()