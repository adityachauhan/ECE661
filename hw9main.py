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
    savepath = os.path.join(top_dir, out_dir)

    img1 = readImgCV(img_dir+'/'+img_type+'_1.jpeg')
    img2 = readImgCV(img_dir+'/'+img_type+'_2.jpeg')

    h1,w1,c1=img1.shape

    pts1 = config['PARAMETERS'][img_type+'1']
    pts2 = config['PARAMETERS'][img_type+'2']
    pts1 = str2np(pts1)
    pts2 = str2np(pts2)

    all_id=np.arange(len(pts1))
    comb_img=plot_inliers_outliers(img1, img2,all_id,pts1, pts2)
    save_img("manual_correspondence.jpg",savepath,comb_img)
    # pltshow(comb_img)

    pts1_norm, T1 = normPts(pts1)
    pts2_norm, T2 = normPts(pts2)
    F = calc_F(pts1_norm, pts2_norm, T1, T2)
    print(F, np.linalg.matrix_rank(F))

    F_LM = rearrange(F, 'c h -> (c h)')
    F_LM = least_squares(FCostFun, F_LM, method='lm', args=(pts1, pts2))
    print(F_LM.nfev)
    F_LM = F_LM.x
    F_LM = rearrange(F_LM,'(c h) -> c h', c=3,h=3)
    F_LM = cond_F(F_LM)
    F_LM = F_LM/F_LM[2,2]
    print(F_LM, np.linalg.matrix_rank(F_LM))

    eL_LM, eR_LM, Ex_LM = findEpipole(F_LM)
    print(eL_LM, eR_LM, Ex_LM)
    P1_LM, P2_LM = getP(F_LM, eR_LM, Ex_LM)
    print(P1_LM, P2_LM)

    H1,H2 = get_homographies(h1,w1, pts1, pts2, eL_LM, eR_LM)
    print(H1,H2)

    h_img1_rect = applyHomography(img1, H1)
    h_img2_rect = applyHomography(img2, H2)
    save_img("rectified_img1.jpg", savepath, h_img1_rect)
    save_img("rectified_img2.jpg", savepath, h_img2_rect)
    # pltshow(h_img1_rect)
    # pltshow(h_img2_rect)

    h_img1_rect_gray = bgr2gray(h_img1_rect)
    h_img2_rect_gray = bgr2gray(h_img2_rect)

    edge1 = cannyEdge(h_img1_rect, True, 300,200,5)
    edge2 = cannyEdge(h_img2_rect, True, 300,200,5)
    save_img_v2("edge_map1.jpg", savepath, edge1)
    save_img_v2("edge_map2.jpg", savepath, edge2)
    # cv2show(edge1,'edge1')
    # cv2show(edge2,'edge2')
    corrs = get_corrs(edge1, edge2, 2)

    corrs_idx = ssd_corrs(corrs[:,0], corrs[:,1], h_img1_rect_gray, h_img2_rect_gray, win_size=3)
    corrImg = plot_corrs(h_img1_rect, h_img2_rect, corrs, corrs_idx, 150)
    save_img("auto_correspondence.jpg", savepath, corrImg)
    # pltshow(corrImg)

    manual_corrs_world_pts = Triangulate(pts1, pts2, P1_LM, P2_LM)
    auto_corrs_worls_pts = Triangulate(corrs[:,0], corrs[:,1], P1_LM, P2_LM)

    plot_3D_point_cloud(manual_corrs_world_pts, savepath, "manual_point_cloud.jpg")
    plot_3D_point_cloud(auto_corrs_worls_pts, savepath,  "auto_point_cloud.jpg")
    plot_3D_projection(manual_corrs_world_pts, savepath, "manual_world_pts.jpg")
    plot_3D_projection(auto_corrs_worls_pts, savepath, "auto_world_pts.jpg")



if __name__ == '__main__':
    main()