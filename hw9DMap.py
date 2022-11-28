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
    dis_data_dir=config['PARAMETERS']['dis_data_dir']
    out_dir=config['PARAMETERS']['out_dir']
    dis_img=config['PARAMETERS']['dis_img']
    dis_map=config['PARAMETERS']['dis_map']
    img_dir = os.path.join(top_dir, dis_data_dir, dis_data_dir)
    savepath = os.path.join(top_dir, out_dir)
    img2 = readImgCV(img_dir + '/' + dis_img + '2.png')
    img1 = readImgCV(img_dir + '/' + dis_img + '6.png')
    disp2 = readImgCV(img_dir + '/' + dis_map + '2.png')
    disp1 = readImgCV(img_dir + '/' + dis_map + '6.png')
    img1_gray = bgr2gray(img1)
    img2_gray = bgr2gray(img2)
    win_size=51
    gt_disp1,d_max1 = get_dmax(disp1)
    gt_disp2,d_max2 = get_dmax(disp2)
    print(d_max1, d_max2)
    dmap1 = apply_census_l2r(img1_gray, img2_gray, d_max1, win_size)
    dmap2 = apply_census_r2l(img2_gray, img1_gray, d_max2, win_size)
    view_dmap(dmap1, "disp1_"+str(win_size)+".jpg", savepath)
    view_dmap(dmap2, "disp2_"+str(win_size)+".jpg", savepath)
    e1 = error_disp(dmap1, gt_disp1,"error1_"+str(win_size)+".jpg", savepath)
    e2 = error_disp(dmap2, gt_disp2,"error2_"+str(win_size)+".jpg", savepath)
    print(e1, e2)

if __name__ == '__main__':
    main()