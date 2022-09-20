import configparser
import os
import matplotlib.pyplot as plt
import numpy as np
from utility import *
from tqdm import tqdm

from vision import *

config = configparser.ConfigParser()
config.read('hw3config.txt')

def main():
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    x_pts = config['PARAMETERS']['ol_pts1']
    x_img = readImgCV(x_img_path)

    ho, wo, co = x_img.shape
    print(ho,wo,co)
    x_pts = str2np(x_pts)

    #############################################################################################
    ##################     Code block to remove the projective homography using VL      #########
    #############################################################################################

    hvl, vl = vanishing_line_homography(x_pts)
    hvlinv=np.linalg.inv(hvl)
    hPrime = xpeqhx(0, 0, hvl)
    wpa1, hpa1 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(0, ho, hvl)
    wpa2, hpa2 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(wo, 0, hvl)
    wpa3, hpa3 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(wo, ho, hvl)
    wpa4, hpa4 = hPrime[0], hPrime[1]
    wpa = max(wpa1, max(wpa2, max(wpa3, wpa4)))
    hpa = max(hpa1, max(hpa2, max(hpa3, hpa4)))
    empty_img = np.ones((int(hpa), int(wpa), 3), np.int32)
    print(empty_img.shape)

    for c in tqdm(range(int(wpa))):
        for r in range(int(hpa)):
            X_prime=xpeqhx(c,r,hvlinv)
            # print(X_prime)
            if np.ceil(X_prime[1]) < ho and np.ceil(X_prime[0]) < wo and X_prime[0]>=0 and X_prime[1]>=0:
                empty_img[r][c] = x_img[int(X_prime[1])][int(X_prime[0])]

    plt.imshow(empty_img)
    plt.show()

    #############################################################################################
    ##################    Code block to remove the affine homography using degenate conic #######
    #############################################################################################

    ol_pts1 = config['PARAMETERS']['ol_pts1']
    ol_pts2 = config['PARAMETERS']['ol_pts2']
    ol_pts1 = str2np(ol_pts1)
    ol_pts2 = str2np(ol_pts2)
    pts_array = np.array((ol_pts1,ol_pts2))
    lset = []
    mset = []
    for pts in range(len(pts_array)):
        lset.append(make_line_hc(pts_array[pts][0], pts_array[pts][1]))
        mset.append(make_line_hc(pts_array[pts][1], pts_array[pts][3]))
    L = np.ones((len(lset),2))
    for lines in range(len(lset)):
        L[lines][0] = lset[lines][0] * mset[lines][0]
        L[lines][1] = (lset[lines][0] * mset[lines][1] + lset[lines][1] * mset[lines][0])
    M = np.ones(len(mset))
    for lines in range(len(lset)):
        M[lines] = -(lset[lines][1]*mset[lines][1])
    Linv = np.linalg.inv(L)
    S = np.dot(Linv, M.T)
    print(S)
    Smat = np.ones((2,2))
    Smat[0][0] = S[0]
    Smat[0][1] = Smat[1][0] = S[1]
    Smat[1][1] = 1
    print("Smat",Smat)
    U,D2,UT = np.linalg.svd(Smat)
    print(U,D2,UT)
    print("D2",D2)
    D = np.sqrt(D2)
    D = np.diag(D)
    print("D", D)
    A = np.dot(np.dot(U,D), np.transpose(U))
    # A = np.dot(U,np.dot(D, np.transpose(U)))
    print("A",A)
    ha = np.zeros((3,3))
    ha[0][0] = A[0][0]
    ha[0][1] = A[0][1]
    ha[1][0] = A[1][0]
    ha[1][1] = A[1][1]
    ha[2][2]=1
    ha = np.dot(np.linalg.inv(ha), hvl)
    hainv=np.linalg.inv(ha)
    print(ha)
    hPrime = xpeqhx(0, 0, ha)
    wpa1,hpa1 = hPrime[0],hPrime[1]
    hPrime = xpeqhx(0, ho, ha)
    wpa2,hpa2 = hPrime[0],hPrime[1]
    hPrime = xpeqhx(wo, 0, ha)
    wpa3,hpa3 = hPrime[0],hPrime[1]
    hPrime = xpeqhx(wo, ho, ha)
    wpa4,hpa4 = hPrime[0],hPrime[1]
    wpa = max(wpa1, max(wpa2, max(wpa3,wpa4)))
    hpa = max(hpa1, max(hpa2, max(hpa3,hpa4)))
    empty_img2 = np.ones((int(hpa), int(wpa), 3), np.int32)
    print(empty_img2.shape)
    for c in tqdm(range(int(wpa))):
        for r in range(int(hpa)):
            X_prime = xpeqhx(c, r, hainv)
            if np.ceil(X_prime[1]) < ho and np.ceil(X_prime[0]) < wo and X_prime[0] >= 0 and X_prime[1] >= 0:
                empty_img2[r][c] = x_img[int(X_prime[1])][int(X_prime[0])]
    plt.imshow(empty_img2)
    plt.show()

if __name__=='__main__':
    main()