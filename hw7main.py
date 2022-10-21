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
    # imgPath = training_dir + '/cloudy1.jpg'
    # img = readImgCV(imgPath)
    # hist = LBP(img)
    # print(hist)
    # #
    dummyArray = np.ones((3,3))*255
    dummyArray[1,1]=0
    print(dummyArray)
    val = interpolate(dummyArray)
    print(val)
    # cv2show(img, 'img')
    # for i in range(1):
    #     img = readImgCV(training_data[i])
    #     LBP(img)

if __name__=='__main__':
    main()