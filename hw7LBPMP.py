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
from multiprocessing.pool import Pool

config = configparser.ConfigParser()
config.read('hw7config.txt')
def classify(data):
    histogram = []
    labels = []
    for i in trange(len(data)):
        img = readImgCV(data[i])
        if img is not None:
            name = data[i].split('.')[0].split('/')[-1]
            labels.append(getLabel(name))
            hist = LBP(img)
            histogram.append(hist)
    return (histogram, labels)

def main():
    top_dir = config['PARAMETERS']['top_dir']
    data_dir = config['PARAMETERS']['data_dir']
    training_dir = config['PARAMETERS']['training_dir']
    training_dir = os.path.join(top_dir,data_dir,training_dir)
    testing_dir = config['PARAMETERS']['testing_dir']
    testing_dir = os.path.join(top_dir,data_dir,testing_dir)
    training_data = glob.glob(training_dir+'/*.jpg')
    testing_data = glob.glob(testing_dir+'/*.jpg')
    # chunks = []
    # for i in range(cpu_count):
    #     chunks.append(training_data[i:i+batch_size])
    with Pool() as pool:
        for result in pool.map(classify, training_data):
            print(result)



if __name__ == '__main__':
    main()