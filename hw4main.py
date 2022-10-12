import random

import numpy as np

from vision import *
import configparser
from tqdm import tqdm
from utility import *
import matplotlib.pyplot as plt
import os
import cv2


config = configparser.ConfigParser()
config.read('hw4config.txt')

def haar_pts(img_orig, resize=False):
    # img_orig = readImgCV(img_path)
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray / 255.0
    pts = harrisCornerDetector(img_gray, filter=filter, sigma=1.2)
    pts_set = []
    for _ in range(100):
        pts_set.append(random.randint(1, len(pts[0])-1))
    pts_rem=[]
    for i in pts_set:
        pts_rem.append((pts[1][i], pts[0][i]))
    # print(pts_set)
    for pt in pts_rem:
        cv2.circle(img_orig, (pt[0], pt[1]), radius=3, color=(255,0,0), thickness=-1)
    return pts_rem, img_gray
def main():
    img_path1 = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['img_name1'])
    img_path2 = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['img_name2'])
    img_orig1 = readImgCV(img_path1)
    img_orig2 = readImgCV(img_path2)
    if img_orig2.shape[0] > img_orig1.shape[0]:
        img_orig2 = image_resize(img_orig2, height=img_orig1.shape[0])
    elif img_orig2.shape[0] < img_orig1.shape[0]:
        img_orig1 = image_resize(img_orig1, height=img_orig2.shape[0])
    out_dir = config['PARAMETERS']['output']
    sigma_dir = config['PARAMETERS']['sigma']
    out_dir = os.path.join(out_dir, sigma_dir)
    img_name1 = config['PARAMETERS']['img_name1']
    '''
    Harrision Corner Detector
    '''
    pts1, img_gray1 = haar_pts(img_orig1)
    pts2, img_gray2 = haar_pts(img_orig2)
    combined_img_haar = np.concatenate((img_orig1, img_orig2), axis=1)
    img_name = 'haar_' + img_name1.split('_')[0] + '.png'
    save_img(img_name, out_dir, combined_img_haar)
    # print(img_orig1.shape, img_orig2.shape)
    '''
    SSD Matcher
    '''

    combined_img_ssd = np.concatenate((img_orig1, img_orig2), axis=1)
    val=0
    pts_set_ssd=ssd(pts1, pts2, img_gray1, img_gray2, 5)
    for pt in pts_set_ssd:
        val+=pt[2]
    val = val/len(pts_set_ssd)
    good_ssd_pts = 0
    bad_ssd_pts = 0
    for pt in pts_set_ssd:
        pt1 = (pts1[pt[0]][0], pts1[pt[0]][1])
        pt2 = (pts2[pt[1]][0]+img_gray1.shape[1], pts2[pt[1]][1])
        if pt[2]<val:
            cv2.line(combined_img_ssd, pt1, pt2, color=pick_color(), thickness=1)
            good_ssd_pts+=1
        else: bad_ssd_pts+=1

    img_name = 'ssd_'+img_name1.split('_')[0]+'.png'
    save_img(img_name, out_dir, combined_img_ssd)
    # plt.imshow(combined_img_ssd)
    # plt.show()

    '''
    NCC Matcher
    '''

    combined_img_ncc = np.concatenate((img_orig1, img_orig2), axis=1)
    val=0
    pts_set_ncc = ncc(pts1, pts2, img_gray1, img_gray2, 5)
    for pt in pts_set_ncc:
        val += pt[2]
    val = val / len(pts_set_ncc)
    good_ncc_pts = 0
    bad_ncc_pts = 0
    for pt in pts_set_ncc:
        pt1 = (pts1[pt[0]][0], pts1[pt[0]][1])
        pt2 = (pts2[pt[1]][0] + img_gray1.shape[1], pts2[pt[1]][1])
        if pt[2] < val:
            cv2.line(combined_img_ncc, pt1, pt2, color=pick_color(), thickness=1)
            good_ncc_pts+=1
        else: bad_ncc_pts+=1
    img_name = 'ncc_' + img_name1.split('_')[0] + '.png'
    save_img(img_name, out_dir, combined_img_ncc)
    # plt.imshow(combined_img_ncc)
    # plt.show()
    print("GOOD SSD PTS: ", good_ssd_pts)
    print("GOOD NCC PTS: ", good_ncc_pts)
    print("BAD SSD PTS: ", bad_ssd_pts)
    print("BAD NCC PTS: ", bad_ncc_pts)

    '''
    SIFT Detection and Matching
    '''

    sift_img, pts = sift_matching(img_path1, img_path2)
    img_name = 'sift_' + img_name1.split('_')[0] + '.png'
    save_img(img_name, out_dir, sift_img)
    # plt.imshow(sift_img)
    # plt.show()
if __name__ == "__main__":
    main()