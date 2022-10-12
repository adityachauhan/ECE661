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
    out_dir = config['PARAMETERS']['output']
    savename = 'test'
    plot=1
    # Panorama(img_dir, out_dir, savename, plot)
    img_paths = glob.glob(img_dir+'/*.jpg')
    print(img_paths)
    num_images = len(img_paths)
    homography_list = []
    LMoptim = True
    for i in range(num_images-1):
        impath1=img_dir+str(i)+'.jpg'
        impath2=img_dir+str(i+1)+'.jpg'
        img_orig1=readImgCV(impath1)
        img_orig2=readImgCV(impath2)
        img1 = cv2.imread(impath1,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(impath2,cv2.IMREAD_GRAYSCALE)
        pts1, des1 = SIFTpoints(img1)
        pts2, des2 = SIFTpoints(img2)
        matchingPts1, matchingPts2 = sift_match(pts1,des1,pts2,des2)#flann_matching(pts1,pts2,des1,des2)
        idx_set = RANSAC(matchingPts1, matchingPts2)
        # print(idx_set)
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
        if LMoptim:
            optimizedH = rearrange(optimizedH, 'c h -> (c h)')
            optimization = least_squares(costFun, optimizedH, method='lm', args=(inlier_set1, inlier_set2))
            optimizedH = optimization['x']
            optimizedH=optimizedH/optimizedH[-1]
            optimizedH = rearrange(optimizedH, '(c h) -> c h', c=3,h=3)
            print(optimizedH)
        homography_list.append(optimizedH)
    print(homography_list)
    homography_list=np.array(homography_list)
    # images=[readImgCV(img) for img in img_paths]
    mid_img=num_images//2
    print(mid_img)
    h2m=np.eye(3)
    for k in range(mid_img, len(homography_list)):
        h2m = h2m @ np.linalg.inv(homography_list[k])
        homography_list[k] = h2m
    h2m=np.eye(3)
    for k in range(mid_img-1,-1,-1):
        h2m = h2m @ homography_list[k]
        homography_list[k] = h2m
    homography_list=np.insert(homography_list,mid_img,np.eye(3),0)

    print(homography_list)
    # for i in range(num_images):
    #     impath=img_dir+str(i)+'.jpg'
    #     img=readImgCV(impath)
    #     ho,wo,co=img.shape
    #     h=homography_list[i]
    #     hinv=np.linalg.inv(h)
    #     hPrime = xpeqhx(0, 0, h)
    #     wpa1, hpa1 = hPrime[0], hPrime[1]
    #     hPrime = xpeqhx(0, ho, h)
    #     wpa2, hpa2 = hPrime[0], hPrime[1]
    #     hPrime = xpeqhx(wo, 0, h)
    #     wpa3, hpa3 = hPrime[0], hPrime[1]
    #     hPrime = xpeqhx(wo, ho, h)
    #     wpa4, hpa4 = hPrime[0], hPrime[1]
    #     wpa = max(wpa1, max(wpa2, max(wpa3, wpa4)))
    #     hpa = max(hpa1, max(hpa2, max(hpa3, hpa4)))
    #     print(wpa, hpa)
    #     empty_img = np.ones((int(hpa), int(wpa), 3), np.int32)
    #     for c in tqdm(range(int(wpa))):
    #         for r in range(int(hpa)):
    #             X_prime = xpeqhx(c, r, hinv)
    #             X_prime = X_prime.astype(int)
    #             if X_prime[1] < ho and X_prime[0] < wo and X_prime[0] >= 0 and X_prime[1] >= 0:
    #                 empty_img[r][c] = img[X_prime[1]][X_prime[0]]
    #     plt.imshow(empty_img)
    #     plt.show()

    # for i in range()
    tx=0
    for i in range(mid_img):
        path=img_dir+str(i)+'.jpg'
        img=cv2.imread(path)
        tx=tx+img.shape[1]
    H_t = np.eye(3)
    H_t[0,2]=tx
    height=0
    width=0
    for i in range(num_images):
        path = img_dir + str(i) + '.jpg'
        img = cv2.imread(path)
        height=max(height, img.shape[0])
        width=width+img.shape[1]
    comb=np.zeros((height,width,3), np.uint8)

    for i in range(num_images):
        path = img_dir + str(i) + '.jpg'
        img = cv2.imread(path)
        H=H_t @ homography_list[i]
        comb=pan(comb, img, H)
    cv2.imshow('comb', comb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for i in range(len(matchingPts1)):
    #     if i in idx_set:
    #         cv2.circle(img_orig1, (int(matchingPts1[i][0]), int(matchingPts1[i][1])), radius=3, color=(255, 0, 0), thickness=-1)
    #     else:
    #         cv2.circle(img_orig1, (int(matchingPts1[i][0]), int(matchingPts1[i][1])), radius=3, color=(0,255, 0), thickness=-1)
    # cv2.imshow("img", img_orig1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for i in range(len(matchingPts2)):
    #     if i in idx_set:
    #         cv2.circle(img_orig2, (int(matchingPts2[i][0]), int(matchingPts2[i][1])), radius=3, color=(255, 0, 0),
    #                    thickness=-1)
    #     else:
    #         cv2.circle(img_orig2, (int(matchingPts2[i][0]), int(matchingPts2[i][1])), radius=3, color=(0, 255, 0),
    #                    thickness=-1)
    # cv2.imshow("img", img_orig2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # combined_img = np.concatenate((img_orig1, img_orig2), axis=1)
    # num_points= len(matchingPts1) if len(matchingPts1)>100 else 100
    # matchingPts1 = matchingPts1[:num_points]
    # matchingPts2 = matchingPts2[:num_points]
    # for i in range(num_points):
    #     mp1 = (int(matchingPts1[i][0]), int(matchingPts1[i][1]))
    #     mp2 = (int(matchingPts2[i][0])+img1.shape[1], int(matchingPts2[i][1]))
    #     cv2.line(combined_img, mp1, mp2, color=(0,255,0), thickness=1)
    #
    # for i in range(len(orig_set)):
    #     cv2.circle(img_orig2, (int(orig_set[i][0]), int(orig_set[i][1])), radius=4, color=(255, 0, 0), thickness=-1)
    #     cv2.circle(img_orig2, (int(idx_set[i][0]), int(idx_set[i][1])), radius=4, color=(0, 255, 0), thickness=-1)
    # cv2.imshow("img", img_orig2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()