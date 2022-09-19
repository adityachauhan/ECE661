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
    ol_pts1 = config['PARAMETERS']['ol_pts1']
    ol_pts2 = config['PARAMETERS']['ol_pts2']
    ol_pts3 = config['PARAMETERS']['ol_pts3']
    ol_pts4 = config['PARAMETERS']['ol_pts4']
    ol_pts5 = config['PARAMETERS']['ol_pts5']
    ol_pts6 = config['PARAMETERS']['ol_pts6']

    ol_pts1 = str2np(ol_pts1)
    ol_pts2 = str2np(ol_pts2)
    ol_pts3 = str2np(ol_pts3)
    ol_pts4 = str2np(ol_pts4)
    ol_pts5 = str2np(ol_pts5)
    ol_pts6 = str2np(ol_pts6)
    # print(ol_pts1,ol_pts2,ol_pts3,ol_pts4,ol_pts5)
    l1 = make_line_hc(ol_pts6[0], ol_pts6[1])
    # l1 = l1 / max(l1)
    m1 = make_line_hc(ol_pts6[1], ol_pts6[3])
    # m1 = m1 / max(m1)
    l2 = make_line_hc(ol_pts6[1], ol_pts6[3])
    # l2 = l2 / max(l2)
    m2 = make_line_hc(ol_pts6[3], ol_pts6[2])
    # m2 = m2 / max(m2)
    l3 = make_line_hc(ol_pts6[3], ol_pts6[2])
    # l3 = l3 / max(l3)
    m3 = make_line_hc(ol_pts6[2], ol_pts6[0])
    # m3 = m3 / max(m3)
    l4 = make_line_hc(ol_pts6[2], ol_pts6[0])
    # l4 = l4 / max(l4)
    m4 = make_line_hc(ol_pts6[0], ol_pts6[1])
    # m4 = m4 / max(m4)
    l5 = make_line_hc(ol_pts5[0], ol_pts5[3])
    # l5 = l5 / max(l5)
    m5 = make_line_hc(ol_pts5[1], ol_pts5[2])
    # m5 = m5 / max(m5)
    l6 = make_line_hc(ol_pts6[0], ol_pts6[3])
    # l5 = l5 / max(l5)
    m6 = make_line_hc(ol_pts6[1], ol_pts6[2])
    # m5 = m5 / max(m5)

    # print(l1,m1, l2, m2,l3,m3,l4,m4,l5,m5)
    L = np.ones((5,6))
    L[0][0] = l1[0] * m1[0]
    L[0][1] = (l1[0] * m1[1] + l1[1] * m1[0]) / 2
    L[0][2] = l1[1] * m1[1]
    L[0][3] = (l1[0] * m1[2] + l1[2] * m1[0]) / 2
    L[0][4] = (l1[1] * m1[2] + l1[2] * m1[1]) / 2
    L[0][5] = l1[2] * m1[2]

    L[1][0] = l2[0] * m2[0]
    L[1][1] = (l2[0] * m2[1] + l2[1] * m2[0]) / 2
    L[1][2] = l2[1] * m2[1]
    L[1][3] = (l2[0] * m2[2] + l2[2] * m2[0]) / 2
    L[1][4] = (l2[1] * m2[2] + l2[2] * m2[1]) / 2
    L[1][5] = l2[2] * m2[2]

    L[2][0] = l3[0] * m3[0]
    L[2][1] = (l3[0] * m3[1] + l3[1] * m3[0]) / 2
    L[2][2] = l3[1] * m3[1]
    L[2][3] = (l3[0] * m3[2] + l3[2] * m3[0]) / 2
    L[2][4] = (l3[1] * m3[2] + l3[2] * m3[1]) / 2
    L[2][5] = l3[2] * m3[2]

    L[3][0] = l4[0] * m4[0]
    L[3][1] = (l4[0] * m4[1] + l4[1] * m4[0]) / 2
    L[3][2] = l4[1] * m4[1]
    L[3][3] = (l4[0] * m4[2] + l4[2] * m4[0]) / 2
    L[3][4] = (l4[1] * m4[2] + l4[2] * m4[1]) / 2
    L[3][5] = l4[2] * m4[2]

    L[4][0] = l5[0] * m5[0]
    L[4][1] = (l5[0] * m5[1] + l5[1] * m5[0]) / 2
    L[4][2] = l5[1] * m5[1]
    L[4][3] = (l5[0] * m5[2] + l5[2] * m5[0]) / 2
    L[4][4] = (l5[1] * m5[2] + l5[2] * m5[1]) / 2
    L[4][5] = l5[2] * m5[2]

    # L[5][0] = l6[0] * m6[0]
    # L[5][1] = (l6[0] * m6[1] + l6[1] * m6[0]) / 2
    # L[5][2] = l6[1] * m6[1]
    # L[5][3] = (l6[0] * m6[2] + l6[2] * m6[0]) / 2
    # L[5][4] = (l6[1] * m6[2] + l6[2] * m6[1]) / 2
    # L[5][5] = l6[2] * m6[2]

    lu,ld,lv=np.linalg.svd(L)
    # print(np.argmin(ld))
    # print(lu,ld,lv)
    # print("++++++++++")
    S = lv[np.argmin(ld)]
    S = S/S[-1]

    # print("L",L)
    # M = np.ones(5)*-1
    # M[0] = -(l1[2]*m1[2])
    # M[1] = -(l2[2]*m2[2])
    # M[2] = -(l3[2]*m3[2])
    # M[3] = -(l4[2]*m4[2])
    # M[4] = -(l5[2]*m5[2])
    # print("M",M)
    # Linv = np.linalg.inv(L)
    # S = np.dot(Linv, M)
    # print("S",S)
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
    A = np.dot(np.dot(U,D), U.T)
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
    # ha=np.linalg.inv(ha)
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
    print(wpa,hpa)
    empty_img = np.ones((int(hpa), int(wpa), 3))
    # empty_img = np.ones((2000,2000, 3))
    empty_img=empty_img.astype(int)
    print(empty_img.shape)
    for c in tqdm(range(int(wo))):
        for r in range(int(ho)):
            X_prime = xpeqhx(c, r, ha)
            # print(X_prime)
            if np.ceil(X_prime[1]) < hpa and np.ceil(X_prime[0]) <wpa and X_prime[0] >= 0 and X_prime[1] >= 0:
                # empty_img[int(np.floor(X_prime[1])), int(np.floor(X_prime[0]))] = x_img[r][c]
                # empty_img[int(np.floor(X_prime[1])), int(np.ceil(X_prime[0]))] = x_img[r][c]
                # empty_img[int(np.ceil(X_prime[1])), int(np.floor(X_prime[0]))] = x_img[r][c]
                # empty_img[int(np.ceil(X_prime[1])), int(np.ceil(X_prime[0]))] = x_img[r][c]
                # empty_img[int(X_prime[1]), int(X_prime[0])] = x_img[r][c]
                empty_img[int(X_prime[1]), int(X_prime[0])] = x_img[r][c]
                # empty_img[r][c] = x_img[int(X_prime[1])][int(X_prime[0])]
            # else: print(X_prime)
    plt.imshow(empty_img)
    plt.show()

if __name__=='__main__':
    main()