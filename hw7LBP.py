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
    histograms_train = []
    labels_train = []
    histograms_test = []
    labels_test = []
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

    classify(histograms_train, labels_train, histograms_test, labels_test)



if __name__=='__main__':
    main()