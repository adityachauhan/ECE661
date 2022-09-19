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
    # wPrime = xpeqhx(wo,0,hvl)
    # wp = wPrime[0]
    # hPrime = xpeqhx(wo,ho,hvl)
    # hp=hPrime[1]
    # print(hp,wp,cp)
    # empty_img = np.ones((int(hp), int(wp), 3), np.int32)
    # empty_img = np.ones((6000,6000, 3), np.int32)
    # print(empty_img.shape)
    # for c in tqdm(range(wo)):
    #     for r in range(ho):
    #         X_prime=xpeqhx(c,r,hvl)
    #         # print(X_prime)
    #         if np.ceil(X_prime[1]) < hp and np.ceil(X_prime[0]) < wp and X_prime[0]>=0 and X_prime[1]>=0:
    #             empty_img[int(np.floor(X_prime[1])),int(np.floor(X_prime[0]))] = x_img[r][c]
    #             empty_img[int(np.floor(X_prime[1])),int(np.ceil(X_prime[0]))] = x_img[r][c]
    #             empty_img[int(np.ceil(X_prime[1])),int(np.floor(X_prime[0]))] = x_img[r][c]
    #             empty_img[int(np.ceil(X_prime[1])),int(np.ceil(X_prime[0]))] = x_img[r][c]
    #             # empty_img[int(X_prime[1]),int(X_prime[0])] = x_img[r][c]
    #
    # plt.imshow(empty_img)
    # plt.show()

    # x_prime_pts = []
    # for i in range(len(x_pts)):
    #     temp_pts = point2point(x_pts[i], hvl)
    #     x_prime_pts.append(temp_pts)
    ol_pts1 = config['PARAMETERS']['ol_pts1']
    ol_pts2 = config['PARAMETERS']['ol_pts2']
    ol_pts1 = str2np(ol_pts1)
    ol_pts2 = str2np(ol_pts2)
    l1=make_line_hc(ol_pts1[0], ol_pts1[3])
    l1=l1/l1[-1]
    m1=make_line_hc(ol_pts1[1], ol_pts1[2])
    m1=m1/m1[-1]
    l2=make_line_hc(ol_pts2[0], ol_pts2[3])
    l2=l2/l2[-1]
    m2=make_line_hc(ol_pts2[1], ol_pts2[2])
    m2=m2/m2[-1]
    print(l1,m1, l2, m2)
    L = np.ones((2,2))
    L[0][0] = l1[0]*m1[0]
    L[0][1] = l1[0]*m1[1] + l1[1]*m1[0]
    L[1][0] = l2[0]*m2[0]
    L[1][1] = l2[0]*m2[1] + l2[1]*m2[0]
    print(L)
    M = np.ones(2)
    M[0] = -(l1[1]*m1[1])
    M[1] = -(l2[1]*m2[1])
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
    # ha = np.dot(np.linalg.inv(hvl), ha)
    # ha = np.dot(hvl, np.linalg.inv(ha))
    # ha=np.linalg.inv(ha)
    print(ha)
    # wPrime = xpeqhx(wp, 0, ha)
    # wpa = wPrime[0]
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
    # print(wpa,hpa)
    empty_img2 = np.ones((int(hpa), int(wpa), 3), np.int32)
    # empty_img2 = np.ones((15000,15000, 3), np.int32)
    print(empty_img2.shape)
    for c in tqdm(range(int(wo))):
        for r in range(int(ho)):
            X_prime = xpeqhx(c, r, ha)
            # print(X_prime)
            if np.ceil(X_prime[1]) < hpa-1 and np.ceil(X_prime[0]) < wpa-1 and X_prime[0] >= 0 and X_prime[1] >= 0:
                empty_img2[int(np.floor(X_prime[1])), int(np.floor(X_prime[0]))] = x_img[r][c]
                empty_img2[int(np.floor(X_prime[1])), int(np.ceil(X_prime[0]))] = x_img[r][c]
                empty_img2[int(np.ceil(X_prime[1])), int(np.floor(X_prime[0]))] = x_img[r][c]
                empty_img2[int(np.ceil(X_prime[1])), int(np.ceil(X_prime[0]))] = x_img[r][c]
                # empty_img2[int(X_prime[1]), int(X_prime[0])] = x_img[r][c]
                # empty_img2[r][c] = x_img[int(X_prime[1])][int(X_prime[0])]
            # else: print(X_prime)
    plt.imshow(empty_img2)
    plt.show()

if __name__=='__main__':
    main()