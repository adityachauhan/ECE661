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

config = configparser.ConfigParser()
config.read('hw7config.txt')

def main():
    top_dir = config['PARAMETERS']['top_dir']
    data_dir = config['PARAMETERS']['data_dir']
    training_dir = config['PARAMETERS']['training_dir']
    training_dir = os.path.join(top_dir,data_dir,training_dir)
    testing_dir = config['PARAMETERS']['testing_dir']
    testing_dir = os.path.join(top_dir,data_dir,testing_dir)
    training_data = glob.glob(training_dir+'/*.jpg')
    testing_data = glob.glob(testing_dir+'/*.jpg')
    output = config['PARAMETERS']['output']


    classes = config['PARAMETERS']['classes']
    classes = classes.split(',')
    histograms_train = []
    labels_train = []
    histograms_test = []
    labels_test = []

    plotting_dir = config['PARAMETERS']['plotting_dir']
    plotting_dir = os.path.join(top_dir, data_dir, plotting_dir)
    plotting_data = glob.glob(plotting_dir+'/*.jpg')
    histograms_plotting = []
    labels_plotting = []

    for i in trange(len(training_data)):
        img = readImgCV(training_data[i])
        if img is not None:
            name = training_data[i].split('.')[0].split('/')[-1]
            labels_train.append(getLabel(name))
            hist = LBP(img)
            histograms_train.append(hist)
        else:
            print("Cant open training file: ", training_data[i])


    for i in trange(len(testing_data)):
        img = readImgCV(testing_data[i])
        if img is not None:
            name = testing_data[i].split('.')[0].split('/')[-1]
            labels_test.append(getLabel(name))
            hist = LBP(img)
            histograms_test.append(hist)
        else:
            print("Can't open testing file: ", testing_data[i])

    for i in trange(len(plotting_data)):
        img = readImgCV(plotting_data[i])
        if img is not None:
            name = plotting_data[i].split('.')[0].split('/')[-1]
            hist_path = os.path.join(output, name+"_LBP_hist.png")
            labels_plotting.append(getLabel(name))
            hist = LBP(img)
            print(hist)
            plthist(hist, np.arange(9), hist_path)
            histograms_plotting.append(hist)
        else:
            print("Can't open testing file: ", plotting_data[i])


    conf_plot_path = os.path.join(output, 'LBPConfMat.png')
    classify(histograms_train, labels_train, histograms_test, labels_test, conf_plot_path, classes)



if __name__=='__main__':
    main()