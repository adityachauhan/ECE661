import configparser
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utility import *
from vision import *

config = configparser.ConfigParser()
config.read('hw3config.txt')

def main():
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])

    x_img = readImgCV(x_img_path)

    ho, wo, co = x_img.shape
    print(ho,wo,co)
    ol_pts1 = config['PARAMETERS']['ol_pts1']
    ol_pts2 = config['PARAMETERS']['ol_pts2']
    ol_pts3 = config['PARAMETERS']['ol_pts3']
    ol_pts4 = config['PARAMETERS']['ol_pts4']
    ol_pts5 = config['PARAMETERS']['ol_pts5']

    ol_pts1 = str2np(ol_pts1)
    ol_pts2 = str2np(ol_pts2)
    ol_pts3 = str2np(ol_pts3)
    ol_pts4 = str2np(ol_pts4)
    ol_pts5 = str2np(ol_pts5)

    pts_array = np.array((ol_pts1,ol_pts2,ol_pts3,ol_pts4,ol_pts5))
    lset = []
    mset = []
    for pts in range(len(pts_array)):
        lset.append(make_line_hc(pts_array[pts][0], pts_array[pts][1]))
        mset.append(make_line_hc(pts_array[pts][1], pts_array[pts][3]))
        lset.append(make_line_hc(pts_array[pts][1], pts_array[pts][3]))
        mset.append(make_line_hc(pts_array[pts][3], pts_array[pts][2]))
        lset.append(make_line_hc(pts_array[pts][3], pts_array[pts][2]))
        mset.append(make_line_hc(pts_array[pts][2], pts_array[pts][0]))
        lset.append(make_line_hc(pts_array[pts][2], pts_array[pts][0]))
        mset.append(make_line_hc(pts_array[pts][0], pts_array[pts][1]))

    lset=np.array(lset)
    mset=np.array(mset)
    print("l here")
    print(lset.shape,lset)
    print("m here")
    print(mset.shape,mset)
    l1 = make_line_hc(ol_pts1[0], ol_pts1[1])
    m1 = make_line_hc(ol_pts1[1], ol_pts1[3])
    print(l1,m1)
    print("end")

    L = np.ones((len(lset),5))
    for lines in range(len(lset)):
        L[lines][0] = lset[lines][0] * mset[lines][0]
        L[lines][1] = (lset[lines][0] * mset[lines][1] + lset[lines][1] * mset[lines][0]) / 2
        L[lines][2] = lset[lines][1] * mset[lines][1]
        L[lines][3] = (lset[lines][0] * mset[lines][2] + lset[lines][2] * mset[lines][0]) / 2
        L[lines][4] = (lset[lines][1] * mset[lines][2] + lset[lines][2] * mset[lines][1]) / 2

    print(L.shape,L)

    M = np.ones(len(mset))*-1
    for lines in range(len(lset)):
        M[lines] = -(lset[lines][2]*mset[lines][2])

    #############################################################################################
    ##########    Code block to remove both projective and  affine homography in one step #######
    #############################################################################################

    S=np.linalg.inv(L.T@L)@L.T@M
    # S = np.dot(Linv, M)
    S=S/max(S)
    print("S",S)
    AAT = np.ones((2,2))
    AAT[0][0]=S[0]
    AAT[0][1]=AAT[1][0]=S[1]/2
    AAT[1][1]=S[2]
    d = np.ones(2)
    d[0]=S[3]/2
    d[1]=S[4]/2
    U,D2,UT = np.linalg.svd(AAT)
    D = np.sqrt(D2)
    D = np.diag(D)
    A = np.dot(np.dot(UT,D), UT.T)
    v = np.dot(np.linalg.inv(A),d)
    ha = np.zeros((3,3))
    ha[0][0] = A[0][0]
    ha[0][1] = A[0][1]
    ha[1][0] = A[1][0]
    ha[1][1] = A[1][1]
    ha[2][0] = v[0]
    ha[2][1] = v[1]
    ha[2][2]=1
    print(ha)
    # hnorm=ha
    hainv=np.linalg.inv(ha)

    hPrime = xpeqhx(0, 0, hainv)
    wpa1,hpa1 = hPrime[0],hPrime[1]
    hPrime = xpeqhx(0, ho, hainv)
    wpa2,hpa2 = hPrime[0],hPrime[1]
    hPrime = xpeqhx(wo, 0, hainv)
    wpa3,hpa3 = hPrime[0],hPrime[1]
    hPrime = xpeqhx(wo, ho, hainv)
    wpa4,hpa4 = hPrime[0],hPrime[1]
    wpa = max(wpa1, max(wpa2, max(wpa3,wpa4)))
    hpa = max(hpa1, max(hpa2, max(hpa3,hpa4)))
    print(wpa,hpa)
    empty_img = np.ones((int(hpa), int(wpa), 3))
    # empty_img = np.ones((2000,2000, 3))
    empty_img=empty_img.astype(int)
    print(empty_img.shape)
    for c in tqdm(range(int(wpa))):
        for r in range(int(hpa)):
            X_prime = xpeqhx(c, r, ha)
            # print(X_prime)
            if np.ceil(X_prime[1]) < ho and np.ceil(X_prime[0]) <wo and X_prime[0] >= 0 and X_prime[1] >= 0:
                empty_img[r][c] = x_img[int(X_prime[1])][int(X_prime[0])]
            # else: print(X_prime)
    plt.imshow(empty_img)
    plt.show()

if __name__=='__main__':
    main()