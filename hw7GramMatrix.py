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
    random_samples = random.sample(range(vgg_channels*vgg_channels), number_gram_samples)
    labels_train=[]
    labels_test=[]
    G_train_set=[]
    G_test_set=[]

    plotting_dir = config['PARAMETERS']['plotting_dir']
    plotting_dir = os.path.join(top_dir, data_dir, plotting_dir)
    plotting_data = glob.glob(plotting_dir + '/*.jpg')
    labels_plotting = []

    output = config['PARAMETERS']['output']


    model_path = os.path.join(top_dir, config['PARAMETERS']['vgg_model'])
    for i in trange(len(training_data)):
        img = readImgCV(training_data[i])
        if img is not None:
            name = training_data[i].split('.')[0].split('/')[-1]
            labels_train.append(getLabel(name))
            feature = vgg_feature_extractor(img, model_path)
            feature = rearrange(feature, 'c h w -> (h w) c')
            G = feature.T @ feature
            G = rearrange(G, 'c h -> (c h)')[random_samples]
            G_train_set.append(G)
        else:
            print("Cant open training file: ", training_data[i])

    for i in trange(len(testing_data)):
        img = readImgCV(testing_data[i])
        if img is not None:
            name = testing_data[i].split('.')[0].split('/')[-1]
            labels_test.append(getLabel(name))
            feature = vgg_feature_extractor(img, model_path)
            feature = rearrange(feature, 'c h w -> (h w) c')
            G = feature.T @ feature
            G = rearrange(G, 'c h -> (c h)')[random_samples]
            G_test_set.append(G)
        else:
            print("Cant open testing file: ", testing_data[i])

    for i in trange(len(plotting_data)):
        img = readImgCV(plotting_data[i])
        if img is not None:
            name = plotting_data[i].split('.')[0].split('/')[-1]
            labels_plotting.append(getLabel(name))
            feature = vgg_feature_extractor(img, model_path)
            feature = rearrange(feature, 'c h w -> (h w) c')
            G = feature.T @ feature
            img_name = name+"_gram_matrix.png"
            cv2show(G, img_name)
            save_img_v2(img_name, output, G)
        else:
            print("Can't open testing file: ", plotting_data[i])

    conf_plot_path = os.path.join(output, 'GramMatrixConfMat.png')
    classify(G_train_set, labels_train, G_test_set, labels_test, conf_plot_path, classes)




if __name__=='__main__':
    main()