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
config.read('hw5config.txt')

def main():
    img_dir = config['PARAMETERS']['top_dir']
    img_type = config['PARAMETERS']['img_type']
    out_dir = config['PARAMETERS']['output']
    num_images = int(config['PARAMETERS']['img_count'])
    img_dir=os.path.join(img_dir, img_type)
    img_dir=img_dir+'/'
    homography_list = []
    mode = config['PARAMETERS']['mode']
    if mode=='scipyLM':
        out_dir=os.path.join(out_dir, 'scipyLM')
    if mode == 'noLM':
        out_dir=os.path.join(out_dir, 'NonLM')
    if mode == 'LM':
        out_dir=os.path.join(out_dir, 'LM')


    for i in range(num_images-1):
        impath1=img_dir+str(i)+'.jpg'
        impath2=img_dir+str(i+1)+'.jpg'
        outImgpath=img_type+'_'+str(i)+'_'+str(i+1)+'_plot.jpg'
        img_orig1=readImgCV(impath1)
        img_orig2=readImgCV(impath2)
        img1 = cv2.imread(impath1,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(impath2,cv2.IMREAD_GRAYSCALE)
        pts1, des1 = SIFTpoints(img1)
        pts2, des2 = SIFTpoints(img2)
        matchingPts1, matchingPts2 = sift_match(pts1,des1,pts2,des2)
        idx_set, costRansac = RANSAC(matchingPts1, matchingPts2)
        print("RANSAC Cost for img %d and %d is %f" % (i, i+1, sum(costRansac)/len(costRansac)))
        plotName = os.path.join(out_dir, str(i)+str(i+1)+'ransac_cost_plot.jpg')
        plotCost(costRansac, plotName)
        inlier_outlier_plot=plot_inliers_outliers(img_orig1, img_orig2, idx_set, matchingPts1, matchingPts2)
        save_img(outImgpath, out_dir, inlier_outlier_plot)
        inlier_set1 = []
        outlier_set1 = []
        inlier_set2 = []
        outlier_set2 = []
        for j in range(len(matchingPts1)):
            if j in idx_set:
                inlier_set1.append(matchingPts1[j])
                inlier_set2.append(matchingPts2[j])
            else:
                outlier_set1.append(matchingPts1[j])
                outlier_set2.append(matchingPts2[j])

        optimizedH = hmat_pinv(inlier_set1, inlier_set2)
        if mode=='LM':
            optimizedH = rearrange(optimizedH, 'c h -> (c h)')
            optimization, costLM = LevMar(optimizedH, inlier_set1, inlier_set2)
            print("LM Cost for img %d and %d is %f" % (i, i + 1, sum(costLM) / len(costLM)))
            plotName = os.path.join(out_dir, str(i) + str(i + 1) + 'LM_cost_plot.jpg')
            plotCost(costRansac, plotName)
            optimizedH = optimization
            optimizedH=optimizedH/optimizedH[-1]
            optimizedH = rearrange(optimizedH, '(c h) -> c h', c=3,h=3)
        if mode=='scipyLM':
            optimizedH = rearrange(optimizedH, 'c h -> (c h)')
            optimization = least_squares(costFun, optimizedH, method='lm', args=(inlier_set1, inlier_set2))
            optimizedH = optimization['x']
            optimizedH=optimizedH/optimizedH[-1]
            optimizedH = rearrange(optimizedH, '(c h) -> c h', c=3,h=3)
        homography_list.append(optimizedH)
    homography_list=np.array(homography_list)


    mid_img=num_images//2
    h2m=np.eye(3)
    for k in range(mid_img, len(homography_list)):
        h2m = h2m @ np.linalg.inv(homography_list[k])
        homography_list[k] = h2m
    h2m=np.eye(3)
    for k in range(mid_img-1,-1,-1):
        h2m = h2m @ homography_list[k]
        homography_list[k] = h2m
    homography_list=np.insert(homography_list,mid_img,np.eye(3),0)


    transW=0
    transH=0
    for i in range(mid_img):
        path=img_dir+str(i)+'.jpg'
        img=cv2.imread(path)
        transH=transH+img.shape[0]
        transW=transW+img.shape[1]
    H_t = np.eye(3)
    H_t[0,2]=transW
    # H_t[1,2]=transH
    height=0
    width=0
    for i in range(num_images):
        path = img_dir + str(i) + '.jpg'
        img = cv2.imread(path)
        height=max(height, findmaxmin(img.shape[0], img.shape[1], homography_list[i])[1])
        width=width+img.shape[1]
    comb=np.zeros((height,width,3), np.uint8)

    for i in range(num_images):
        path = img_dir + str(i) + '.jpg'
        img = cv2.imread(path)
        H=H_t@homography_list[i]
        comb=create_panaroma(comb, img, H)

    comb=cv2.cvtColor(comb, cv2.COLOR_BGR2RGB)
    finImgpath = img_type+"_panaroma.jpg"
    save_img(finImgpath, out_dir, comb)


if __name__ == '__main__':
    main()