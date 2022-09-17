import configparser
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vision import *

config = configparser.ConfigParser()
config.read('hw3config.txt')

def main():
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    x_pts = config['PARAMETERS']['l_pts']
    # x_prime_pts = config['PARAMETERS']['l_prime_pts']
    homo_mode = config['PARAMETERS']['homo_mode']
    x_img = readImgCV(x_img_path)
    ho, wo, co = x_img.shape
    print(ho,wo,co)
    x_pts = str2np(x_pts)
    # x_prime_pts=str2np(x_prime_pts)
    hvl, vl = vanishing_line_homography(x_pts)
    wPrime = xpeqhx(wo,0,hvl)
    wp = wPrime[0]
    hPrime = xpeqhx(wo,ho,hvl)
    hp=hPrime[1]
    # print(hp,wp,cp)
    empty_img = np.ones((int(hp), int(wp), 3), np.int32)
    # empty_img = np.ones((6000,6000, 3), np.int32)
    print(empty_img.shape)
    for c in tqdm(range(wo)):
        for r in range(ho):
            X_prime=xpeqhx(c,r,hvl)
            # print(X_prime)
            if np.ceil(X_prime[1]) < hp and np.ceil(X_prime[0]) < wp and X_prime[0]>=0 and X_prime[1]>=0:
                empty_img[int(np.floor(X_prime[1])),int(np.floor(X_prime[0]))] = x_img[r][c]
                empty_img[int(np.floor(X_prime[1])),int(np.ceil(X_prime[0]))] = x_img[r][c]
                empty_img[int(np.ceil(X_prime[1])),int(np.floor(X_prime[0]))] = x_img[r][c]
                empty_img[int(np.ceil(X_prime[1])),int(np.ceil(X_prime[0]))] = x_img[r][c]
                # empty_img[int(X_prime[1]),int(X_prime[0])] = x_img[r][c]

    plt.imshow(empty_img)
    plt.show()

    # x_prime_pts = []
    # for i in range(len(x_pts)):
    #     temp_pts = point2point(x_pts[i], hvl)
    #     x_prime_pts.append(temp_pts)
    ol_pts1 = config['PARAMETERS']['ol_pts1']
    ol_pts2 = config['PARAMETERS']['ol_pts2']
    ol_pts1 = str2np(ol_pts1)
    ol_pts2 = str2np(ol_pts2)
    l12=make_line_hc(ol_pts1[0], ol_pts1[1])
    l13=make_line_hc(ol_pts1[2], ol_pts1[3])
    l43=make_line_hc(ol_pts2[0], ol_pts2[1])
    l42=make_line_hc(ol_pts2[2], ol_pts2[3])
    print(l12,l13, l43, l42)
    L = np.ones((2,2))
    L[0][0] = l12[0]*l13[0]
    L[0][1] = l12[0]*l13[1] + l12[1]*l13[0]
    L[1][0] = l43[0]*l42[0]
    L[1][1] = l43[0]*l42[1] + l43[1]*l42[0]
    print(L)
    M = np.ones(2)
    M[0] = -l12[1]*l13[1]
    M[1] = -l43[1]*l42[1]
    Linv = np.linalg.inv(L)
    S = np.dot(Linv, M)
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
    A = np.dot(U,np.dot(D, np.transpose(U)))
    print("A",A)
    ha = np.zeros((3,3))
    ha[0][0] = A[0][0]
    ha[0][1] = A[0][1]
    ha[1][0] = A[1][0]
    ha[1][1] = A[1][1]
    ha[2][2]=1
    ha=np.linalg.inv(ha)
    # ha = np.matmul(ha, hvl)
    print(ha)
    # wPrime = xpeqhx(wp, 0, ha)
    # wpa = wPrime[0]
    # hPrime = xpeqhx(wp, hp, ha)
    # hpa = hPrime[1]
    empty_img2 = np.ones((int(hp), int(wp), 3), np.int32)
    # empty_img2 = np.ones((15000,15000, 3), np.int32)
    print(empty_img2.shape)
    for c in tqdm(range(int(wp))):
        for r in range(int(hp)):
            X_prime = xpeqhx(c, r, ha)
            # print(X_prime)
            if np.ceil(X_prime[1]) < hp and np.ceil(X_prime[0]) < wp and X_prime[0] >= 0 and X_prime[1] >= 0:
                # empty_img2[int(np.floor(X_prime[1])), int(np.floor(X_prime[0]))] = empty_img[r][c]
                # empty_img2[int(np.floor(X_prime[1])), int(np.ceil(X_prime[0]))] = empty_img[r][c]
                # empty_img2[int(np.ceil(X_prime[1])), int(np.floor(X_prime[0]))] = empty_img[r][c]
                # empty_img2[int(np.ceil(X_prime[1])), int(np.ceil(X_prime[0]))] = empty_img[r][c]
                empty_img2[int(X_prime[1]), int(X_prime[0])] = x_img[r][c]
                # empty_img2[r][c] = empty_img[int(X_prime[1])][int(X_prime[0])]
            # else: print(X_prime)
    plt.imshow(empty_img2)
    plt.show()

if __name__=='__main__':
    main()