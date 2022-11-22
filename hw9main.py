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
    pts1 = config['PARAMETERS'][img_type+'1']
    pts2 = config['PARAMETERS'][img_type+'2']
    pts1 = str2np(pts1)
    pts2 = str2np(pts2)
    # all_id=np.arange(len(pts1))
    # comb_img=plot_inliers_outliers(img1, img2,all_id,pts1, pts2)
    # cv2show(comb_img, 'img')
    pts1_norm, T1 = normPts(pts1)
    pts2_norm, T2 = normPts(pts2)
    F = calc_F(pts1_norm, pts2_norm, T1, T2)
    print(F, np.linalg.matrix_rank(F))
    eL,eR,Ex=findEpipole(F)
    print(eL, eR, Ex)
    P1, P2 = getP(F, eR, Ex)
    print(P1, P2)









if __name__ == '__main__':
    main()