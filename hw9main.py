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
from einops import rearrange
config = configparser.ConfigParser()
config.read('hw9config.txt')

def main():
    top_dir=config['PARAMETERS']['top_dir']
    data_dir=config['PARAMETERS']['data_dir']
    out_dir=config['PARAMETERS']['out_dir']
    img_type=config['PARAMETERS']['img_type']
    img_dir = os.path.join(top_dir, data_dir)
    img1 = readImgCV(img_dir+'/'+img_type+'_1.jpeg')
    img2 = readImgCV(img_dir+'/'+img_type+'_2.jpeg')
    gray_img1 = bgr2gray(img1)
    gray_img2 = bgr2gray(img2)
    pts1, des1 = SIFTpoints_v2(gray_img1)
    pts2, des2 = SIFTpoints_v2(gray_img2)
    mp1,mp2 = flann_matching(img1, img2,pts1, des1, pts2, des2)
    all_idx = np.arange(len(mp1))
    # idx_set, costRansac = RANSAC(mp1, mp2)
    inlier_plot = plot_inliers_outliers(img1, img2, all_idx, mp1, mp2, True)
    cv2show(inlier_plot, 'img')








if __name__ == '__main__':
    main()