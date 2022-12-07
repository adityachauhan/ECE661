import glob
import os

import random
from scipy.optimize import least_squares

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from einops import rearrange, repeat, reduce
from tqdm import trange, tqdm
from scipy.signal import convolve2d
import sys
import BitVector
from sklearn import svm
from sklearn.metrics import confusion_matrix, \
    ConfusionMatrixDisplay, accuracy_score
from HW7Auxilliary.vgg import VGG19
from skimage import transform
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def readImgCV(path):
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else: return None
    return img

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def cvrt2homo(pt):
    return np.append(pt, 1)

def gauss_blur(img, kernel_size=5):
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur
def cv2show(img, name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(img, size):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img
def pltshow(img):
    plt.imshow(img)
    plt.show()
def plotCost(cost, name):
    plt.figure()
    plt.scatter(range(len(cost)), np.array(cost), s=80, edgecolors='black', c='red', label='data')
    plt.xlabel('Iterations')
    plt.xlabel('Cost')
    plt.title("Cost VS Iterations")
    plt.legend()
    plt.savefig(name)

def str2np(s):
    x_prime_pts = s.split(',')
    x_prime_pts = [int(val) for val in x_prime_pts]
    x_prime_pts = np.array(x_prime_pts)
    x_prime_pts = rearrange(x_prime_pts, '(c h)-> c h ', c=len(x_prime_pts)//2, h=2)
    return x_prime_pts

def pts2hc(pts):
    pts_hc = np.ones((len(pts),3))
    pts_hc[:,:2] = pts
    return pts_hc
def save_img(name, path, img):
    img_path = os.path.join(path, name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, img)

def save_img_v2(name, path, img):
    img_path = os.path.join(path, name)
    cv2.imwrite(img_path, img)
def xpeqhx(x,y,h):
    X = np.array((x, y, 1))
    X_prime = np.matmul(h, X)
    X_prime = X_prime / X_prime[2]
    X_prime = X_prime.astype(int)
    return X_prime

def point2point(x,h):
    X = np.array((x[0], x[1], 1))
    X_prime = np.matmul(h, X)
    X_prime = X_prime/X_prime[2]
    X_prime = X_prime.astype(int)
    X_prime = np.array([X_prime[0], X_prime[1]])
    return X_prime
def vanishing_line_homography(x_pts):
    l12 = np.cross(cvrt2homo(x_pts[0]), cvrt2homo(x_pts[1]))
    l34 = np.cross(cvrt2homo(x_pts[2]), cvrt2homo(x_pts[3]))
    l31 = np.cross(cvrt2homo(x_pts[2]), cvrt2homo(x_pts[0]))
    l42 = np.cross(cvrt2homo(x_pts[3]), cvrt2homo(x_pts[1]))
    vpt1 = np.cross(l12, l34)
    vpt1 = vpt1 / vpt1[2]
    vpt2 = np.cross(l31, l42)
    vpt2 = vpt2 / vpt2[2]
    vl = np.cross(vpt1, vpt2)
    h = np.zeros((3,3))
    h[0][0] = 1
    h[1][1] = 1
    h[2][0] = vl[0]/vl[2]
    h[2][1] = vl[1]/vl[2]
    h[2][2] = vl[2]/vl[2]
    return h, vl

def primePtsP2P(pts):
    X=np.ones((len(pts),2))
    # print(X)
    X[0,0]=pts[0,0]
    X[0,1]=pts[0,1]
    X[1,0]=pts[1,0]
    X[1,1]=pts[0,1]
    X[2,0]=pts[0,0]
    X[2,1]=pts[2,1]
    X[3,0]=pts[1,0]
    X[3,1]=pts[2,1]
    return X
def make_line_hc(pts1, pts2):
    l = np.cross(cvrt2homo(pts1), cvrt2homo(pts2))
    return l


class Vision:
    def __init__(self, x, x_prime, homo_mat_size=3):
        self.x = x
        self.x_prime = x_prime
        self.h = np.ones((homo_mat_size,homo_mat_size))
        self.h_mat_size = homo_mat_size

    def calc_homograpy(self, homo_mode):
        if homo_mode=='projective':
            A = np.ones((len(self.x)*2,len(self.x)*2))
            C = np.ones(len(self.x)*2)
            for i in range(len(self.x)):
                A[2*i] = np.array([self.x[i][0], self.x[i][1], 1, 0, 0, 0, -self.x[i][0]*self.x_prime[i][0], -self.x[i][1]*self.x_prime[i][0]])
                A[(2*i)+1] = np.array([0, 0, 0, self.x[i][0], self.x[i][1], 1, -self.x[i][0]*self.x_prime[i][1], -self.x[i][1]*self.x_prime[i][1]])
                C[2*i] = self.x_prime[i][0]
                C[(2*i)+1] = self.x_prime[i][1]
            Ainv = np.linalg.inv(A)
            B = np.dot(Ainv, C)
            B = np.append(B, 1)
            self.h = rearrange(B, '(c h)-> c h',c=self.h_mat_size, h=self.h_mat_size)
        if homo_mode=='affine':
            A = np.ones(((len(self.x)-1) * 2, (len(self.x) -1)* 2))
            C = np.ones((len(self.x) -1) * 2)
            for i in range(len(self.x)-1):
                A[2 * i] = np.array([self.x[i][0], self.x[i][1], 1, 0, 0, 0])
                A[(2 * i) + 1] = np.array([0, 0, 0, self.x[i][0], self.x[i][1], 1])
                C[2 * i] = self.x_prime[i][0]
                C[(2 * i) + 1] = self.x_prime[i][1]
            Ainv = np.linalg.inv(A)
            B = np.dot(Ainv, C)
            B = np.append(B, (0,0,1))
            self.h = rearrange(B, '(c h)-> c h', c=self.h_mat_size, h=self.h_mat_size)
        return self.h


def harrisCornerDetector(img, filter="HAAR", sigma=1.2):
    # if filter=="SOBEL":
    #     gx, gy = Sobel()
    #
    # if filter=="HAAR":
    gx, gy = Haar_Wavelet(sigma)
    win_size = int(np.ceil(5 * sigma)) if np.ceil(5 * sigma) % 2 == 0 else int(np.ceil(5 * sigma)) + 1
    dx = convolve2d(img, gx, mode="same")
    dy = convolve2d(img, gy, mode="same")

    sumK = np.ones((win_size,win_size))
    dx2 = dx**2
    dy2 = dy**2
    dxdy= np.multiply(dx,dy)

    sum_dx2=convolve2d(dx2, sumK,mode="same")
    sum_dy2=convolve2d(dy2, sumK,mode="same")
    sum_dxdy=convolve2d(dxdy, sumK,mode="same")

    detC = sum_dx2*sum_dy2 - (sum_dxdy**2)
    trC = sum_dx2+sum_dy2
    trC2 = trC**2
    k=0.05
    Ratios = detC - k*trC2
    Ratios*(Ratios==maximum_filter(Ratios, footprint=np.ones((5,5))))
    corners = np.where(Ratios>0.01*np.max(Ratios))

    # print(len(corners[0]), len(corners[1]))
    return corners

def get_patch(img, r,c,pad):
    return img[r - pad:r + pad + 1, c - pad:c + pad + 1]
def extarct_patch(pt, win_size, img):
    # print(img.shape)
    assert win_size%2 != 0
    pad = int((win_size-1)/2)
    val_minx = pt[0] - pad
    val_maxx = pt[0] + pad
    val_miny = pt[1] - pad
    val_maxy = pt[1] + pad
    win_minx = val_minx #if val_minx > 0 else 0
    win_maxx = val_maxx #if val_maxx < img.shape[1] else img.shape[1]
    win_miny = val_miny #if val_miny > 0 else 0
    win_maxy = val_maxy #if val_maxy < img.shape[0] else img.shape[0]
    # print(win_minx, win_miny, win_maxx, win_maxy)
    pix_vals = []
    for i in range(win_minx, win_maxx):
        for j in range(win_miny, win_maxy):
            pix_vals.append(img[j,i])
    return pix_vals

def ssd(pts1, pts2, img1, img2, win_size):
    combined_pt_set=[]
    patches1=[]
    patches2=[]
    pad=int((win_size-1)/2)
    img1=cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    img2=cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for pt in pts1:
        patches1.append(extarct_patch(pt, win_size, img1))
    for pt in pts2:
        patches2.append(extarct_patch(pt, win_size, img2))

    for i in range(len(patches1)):
        min_dist =[]
        for j in range(len(patches2)):
            min_dist.append(np.mean((np.array(patches1[i])-np.array(patches2[j]))**2))
        min_val = min(v for v in min_dist if v>0.0)
        # combined_pt_set.append([i,np.argmin(np.array(min_dist)), min_dist[np.argmin(np.array(min_dist))]])
        combined_pt_set.append([i,min_dist.index(min_val), min_dist[min_dist.index(min_val)]])

    return combined_pt_set


def ncc(pts1, pts2, img1, img2, win_size):
    combined_pt_set = []
    patches1 = []
    patches2 = []
    pad = int((win_size - 1) / 2)
    img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for pt in pts1:
        patches1.append(extarct_patch(pt, win_size, img1))
    for pt in pts2:
        patches2.append(extarct_patch(pt, win_size, img2))

    for i in range(len(patches1)):
        min_dist = []
        p1=np.array(patches1[i])
        for j in range(len(patches2)):
            p2 = np.array(patches2[j])
            numerator = np.sum((p1-np.mean(p1))*(p2-np.mean(p2)))
            denomiator= np.sqrt(np.sum((p1-np.mean(p1))**2)*np.sum((p2-np.mean(p2))**2))
            min_dist.append(1-(numerator/(denomiator+1e-6)))
        min_val = min(v for v in min_dist if v > 0.0)
        combined_pt_set.append([i,min_dist.index(min_val), min_dist[min_dist.index(min_val)]])

    return combined_pt_set

'''
Sift code from opencv documentation page.
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
'''
def sift_matching(img_path1, img_path2):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp1, dv1 = sift.detectAndCompute(img1, None)
    kp2, dv2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(dv1,dv2,k=2)

    points = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            points.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3, points
def Sobel():
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return gx, gy

def Haar_Wavelet(sigma):
    size = int(np.ceil(4 * sigma)) if np.ceil(4 * sigma) % 2 == 0 else int(np.ceil(4 * sigma)) + 1
    gx = np.ones((size, size))
    gx = gx.astype(int)
    gx[:, :int(size / 2)] = -1
    gy = gx.T
    return gx, gy


def pick_color():
    return (random.randint(10,250),random.randint(10,250),random.randint(10,250))

def SIFTpoints(img):
    pts_set=[]
    sift = cv2.SIFT_create()
    pts, des = sift.detectAndCompute(img, None)

    for pt in pts:
        pts_set.append([int(pt.pt[0]), int(pt.pt[1])])
    return np.array(pts_set), des


def SIFTpoints_v2(img):
    sift = cv2.SIFT_create()
    pts, des = sift.detectAndCompute(img, None)
    return pts, des

def siftMatching(pts1, des1,pts2,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    mp1=[]
    mp2=[]


def flann_matching(img1, img2,pts1,des1,pts2,des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.7 * n.distance:
    #         matchesMask[i] = [1, 0]
    #
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=cv2.DrawMatchesFlags_DEFAULT)
    #
    # img3 = cv2.drawMatchesKnn(img1, pts1, img2, pts2, matches, None, **draw_params)
    # return img3
    matchingPts1=[]
    matchingPts2=[]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchingPts1.append((pts1[m.queryIdx].pt[0],pts1[m.queryIdx].pt[1]))
            matchingPts2.append((pts2[n.trainIdx].pt[0],pts2[n.trainIdx].pt[1]))
    matchingPts1 = [[int(val[0]), int(val[1])] for val in matchingPts1]
    matchingPts2 = [[int(val[0]), int(val[1])] for val in matchingPts2]
    return matchingPts1, matchingPts2

def hmat_pinv(x, x_prime):
    A = np.ones((len(x) * 2, 8))
    C = np.ones(len(x) * 2)
    for i in range(len(x)):
        A[2 * i] = np.array([x[i][0], x[i][1], 1, 0, 0, 0, -x[i][0] * x_prime[i][0],
                             -x[i][1] * x_prime[i][0]])
        A[(2 * i) + 1] = np.array([0, 0, 0, x[i][0], x[i][1], 1, -x[i][0] * x_prime[i][1],
                                   -x[i][1] * x_prime[i][1]])
        C[2 * i] = x_prime[i][0]
        C[(2 * i) + 1] = x_prime[i][1]

    h = (np.linalg.inv((A.T)@A)@A.T)@C
    h = np.concatenate([h,[1]])
    h = rearrange(h, '(c h)-> c h',c=3, h=3)
    return h

def RANSAC(pts1, pts2):
    p = 0.99
    n = 20
    sigma=4
    delta = 3*sigma
    e = 0.1
    N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
    print(N)
    n_total = len(pts1)
    pts1_hc = np.ones((n_total, 3))
    pts1_hc[:, :-1] = pts1
    pts2_hc = np.ones((n_total, 3))
    pts2_hc[:, :-1] = pts2
    # print(pts1_hc.shape, pts2_hc.shape)
    outliers=True
    cost = []
    while outliers:
        count=0
        prev_size=0
        idx_set=[]
        while count < N:
            randIdx = np.random.randint(n_total, size=n)
            cpts1 = pts1_hc[randIdx,:]
            cpts2 = pts2_hc[randIdx,:]
            cH = hmat_pinv(cpts1, cpts2)
            est_cpts2 = cH @ pts1_hc.T
            est_cpts2 = est_cpts2/est_cpts2[2]
            est_cpts2 = rearrange(est_cpts2, 'c h -> h c')
            error = (pts2_hc[:,:-1] - est_cpts2[:,:-1])**2
            error = error @ np.ones((2,1))
            cost.append(np.linalg.norm(error))
            in_indices = np.where(error <= delta)[0]
            if len(in_indices) > prev_size:
                print("Processing RANSAC....")
                idx_set = in_indices
                if len(idx_set) > (1 - e) * n_total:
                    outliers = False
                    break
                prev_size=len(in_indices)
            count+=1
        if len(idx_set) > (1-e)*n_total:
            outliers=False
        else:
            e*=2
            N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
    print("num inliers found:", len(idx_set))
    return idx_set, cost

#Euclidian distance based matching found the matcher code online
def sift_match(pts1, des1, pts2, des2, tr=3):
    match_pts1 = []
    match_pts2 = []
    eu = np.zeros((len(pts1), len(pts2)))
    for i in trange(len(pts1)):
        for j in range(len(pts2)):
            eu[i, j] = np.linalg.norm(des1[i, :] - des2[j, :])

    eu = eu / np.min(eu)
    dy_threshold = tr
    for i in trange(len(pts1)):
        eu_min = np.min(eu[i, :])
        if eu_min < dy_threshold:
            j_min = np.argmin(eu[i, :])
            match_pts1.append(pts1[i])
            match_pts2.append(pts2[j_min])
    print('number of sift matches found:', len(match_pts1))
    return match_pts1, match_pts2

def costFun(h, pts1, pts2):
    h = rearrange(h, '(c h)-> c h', c=3,h=3)
    n_total = len(pts1)
    pts1_hc = np.ones((n_total, 3))
    pts1_hc[:, :-1] = pts1
    pts2_hc = np.ones((n_total, 3))
    pts2_hc[:, :-1] = pts2
    est_cpts2 = h @ pts1_hc.T
    est_cpts2 = est_cpts2 / est_cpts2[2]
    est_cpts2 = rearrange(est_cpts2, 'c h -> h c')
    X = rearrange(np.array(pts2), 'c h -> (c h)')
    f = rearrange(np.array(est_cpts2[:,:-1]), 'c h -> (c h)')
    error = X-f
    return error

def create_panaroma(curr_img, new_img, H):
    hc,wc,cc = curr_img.shape
    hn,wn,cn = new_img.shape
    Hinv = np.linalg.inv(H)
    pts_hc = []
    for i in range(hc):
        for j in range(wc):
            pts_hc.append([j,i,1])
    pts_hc = np.array(pts_hc)
    est_pts = Hinv @ pts_hc.T
    est_pts = est_pts/est_pts[2,:]
    est_pts = est_pts.T[:,0:2].astype('int')
    idx= np.where(np.logical_and(np.logical_and(est_pts[:,0]>=0,est_pts[:,0]<=wn-1),np.logical_and(est_pts[:,1]>=0,est_pts[:,1]<=hn-1)))
    pts=pts_hc[idx]
    est_pts=est_pts[idx]
    curr_img[pts[:, 1], pts[:, 0]] = new_img[est_pts[:, 1], est_pts[:, 0]]
    return curr_img

def plot_inliers_outliers(img1, img2, idx, mp1, mp2, only_inliers=False):
    comb_img = np.concatenate((img1, img2), axis=1)
    for i in range(len(mp1)):
        p1 = (int(mp1[i][0]), int(mp1[i][1]))
        p2 = (int(mp2[i][0]) + img1.shape[1], int(mp2[i][1]))
        if only_inliers:
            if i in idx:
                cv2.circle(comb_img, p1, radius=3, color=(0, 255, 0), thickness=-1)
                cv2.circle(comb_img, p2, radius=3, color=(0, 255, 0), thickness=-1)
                cv2.line(comb_img, p1, p2, color=(0, 255, 0), thickness=1)
        else:
            if i in idx:
                cv2.circle(comb_img, p1, radius=3, color=(0, 255, 0),thickness=-1)
                cv2.circle(comb_img, p2, radius=3, color=(0, 255, 0),thickness=-1)
                cv2.line(comb_img, p1, p2, color=(0, 255, 0), thickness=1)

            else:
                cv2.circle(comb_img, p1, radius=3, color=(255, 0, 0), thickness=-1)
                cv2.circle(comb_img, p2, radius=3, color=(255, 0, 0),thickness=-1)
                cv2.line(comb_img, p1, p2, color=(255, 0, 0), thickness=1)

    return comb_img

def plot_inliers_outliers_v2(img1, img2, idx, mp1, mp2, only_inliers=False):
    comb_img = np.concatenate((img1, img2), axis=0)
    for i in range(len(mp1)):
        p1 = (int(mp1[i][0]), int(mp1[i][1]))
        p2 = (int(mp2[i][0]), int(mp2[i][1])+ img1.shape[0])
        if only_inliers:
            if i in idx:
                cv2.circle(comb_img, p1, radius=3, color=(0, 255, 0), thickness=-1)
                cv2.circle(comb_img, p2, radius=3, color=(0, 255, 0), thickness=-1)
                cv2.line(comb_img, p1, p2, color=(0, 255, 0), thickness=1)
        else:
            if i in idx:
                cv2.circle(comb_img, p1, radius=3, color=(0, 255, 0),thickness=-1)
                cv2.circle(comb_img, p2, radius=3, color=(0, 255, 0),thickness=-1)
                cv2.line(comb_img, p1, p2, color=(0, 255, 0), thickness=1)

            else:
                cv2.circle(comb_img, p1, radius=3, color=(255, 0, 0), thickness=-1)
                cv2.circle(comb_img, p2, radius=3, color=(255, 0, 0),thickness=-1)
                cv2.line(comb_img, p1, p2, color=(255, 0, 0), thickness=1)

    return comb_img

def Jacobian(pts, h):
    h = rearrange(h, '(c h) -> c h', c=3,h=3)
    J = np.zeros((len(pts)*2, 9))
    for i in range(len(pts)):
        f=h @ np.array([pts[i][0], pts[i][1], 1])
        J[2*i] = np.array([pts[i][0]/f[-1], pts[i][1]/f[-1], 1/f[-1], 0, 0, 0, (-pts[i][0]*f[0])/(f[-1]**2), (-pts[i][1]*f[0])/(f[-1]**2), -f[0]/(f[-1]**2)])
        J[2*i+1] = np.array([0,0,0,pts[i][0]/f[-1], pts[i][1]/f[-1], 1/f[-1], (-pts[i][0]*f[1])/(f[-1]**2), (-pts[i][1]*f[1])/(f[-1]**2), -f[1]/(f[-1]**2)])

    return J

#Followed from the pseudocode by levmar
def LevMar(H, pts1, pts2):
    I = np.identity(9)
    tau = 0.5
    J = Jacobian(pts1, H)
    A = J.T @ J
    mu = tau * np.max(np.diag(A))
    h=H
    k=0
    kmax=100
    cost=[]
    while k<kmax:
        ce = costFun(h, pts1, pts2)
        c = np.linalg.norm(ce)**2
        Jk = Jacobian(pts1, h)
        deltaP = (np.linalg.inv((Jk.T @ Jk) + mu*I) @ Jk.T) @ ce
        Hk=h + deltaP
        cek = costFun(Hk, pts1, pts2)
        ck = np.linalg.norm(cek)**2
        cost.append(ck)
        rho_num = c-ck
        rho_den = deltaP.T@(((mu*I) @ deltaP) + (Jk.T @ ce))
        rho = rho_num/rho_den
        muk = mu * max(1/3, 1-(2*rho - 1)**3)
        mu = muk
        h=Hk
        k+=1

    return h, cost

def findmaxmin(ho,wo,h):
    hPrime = xpeqhx(0, 0, h)
    wpa1, hpa1 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(0, ho, h)
    wpa2, hpa2 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(wo, 0, h)
    wpa3, hpa3 = hPrime[0], hPrime[1]
    hPrime = xpeqhx(wo, ho, h)
    wpa4, hpa4 = hPrime[0], hPrime[1]
    wpa = max(wpa1, max(wpa2, max(wpa3, wpa4)))
    hpa = max(hpa1, max(hpa2, max(hpa3, hpa4)))
    wpe = min(wpa1, min(wpa2, min(wpa3, wpa4)))
    hpe = min(hpa1, min(hpa2, min(hpa3, hpa4)))
    return wpa, hpa, wpe, hpe

def plthist(hist, bins, path=None):
    plt.bar(bins, hist, color='b', width=1, align='center', alpha=1)
    if path is not None:
        plt.savefig(path)
    plt.show()
def histogram(img, bins):
    h,w = img.shape[0],img.shape[1]
    hist = np.zeros(bins)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1
    return hist

def otsu(img, bins, hist):
    p = hist/np.sum(hist)
    ip=np.arange(1,bins+1)
    mu = np.multiply(p,ip)
    mut= p@ip
    sigmab=np.zeros(bins+1)
    for k in range(1, bins):
        w0 = np.sum(p[:k])
        w1 = 1-w0
        if w0 > 0 and w0 < 1 and w1 > 0 and w1 < 1:
            mu0 = np.sum(mu[:k])
            mu1 = (mut-mu0)/w1
            sigmab[k]=(w0*w1*((mu1-mu0)**2))
    ks = np.argmax(sigmab)
    print(ks)
    _,thresh = cv2.threshold(img, ks, 255, cv2.THRESH_BINARY)
    _,thresh_inv = cv2.threshold(img, ks, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh, thresh_inv



def otsu_rgb(img, bins):
    channels = cv2.split(img)
    chcount = len(channels) if len(channels)==3 else 3
    comb = np.zeros(np.append(channels[0].shape, chcount))*255
    combinv = np.zeros(np.append(channels[0].shape, chcount))*255
    for i in range(len(channels)):
        hist = histogram(channels[i], bins)
        thresh, thresh_inv = otsu(channels[i], bins, hist)
        combinv[:,:,i] = thresh_inv
        comb[:,:,i] = thresh
        # comb = cv2.bitwise_and(comb, thresh_inv)
    return comb, combinv

def channel_and(channels):
    img = cv2.bitwise_and(cv2.bitwise_and(channels[:,:,0], channels[:,:,1]), channels[:,:,2])
    return img

def channel_and_v2(c1,c2,c3):
    img = cv2.bitwise_and(cv2.bitwise_and(c1,c2),c3)
    return img


def otsu_texture(img, window_sizes):
    comb = np.zeros(np.append(img.shape, len(window_sizes)))
    for idx in range(len(window_sizes)):
        padding = window_sizes[idx]//2
        temp = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                win = temp[r:r+(2*padding+1),c:c+(2*padding+1)]
                comb[r,c,idx] = np.var(win-np.mean(win))
    comb=comb.astype(np.uint8)
    return comb
def contour(img):
    img = img//255
    cnt = np.zeros(img.shape).astype(np.uint8)
    padding = 1
    temp = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    for r in range(padding,img.shape[0]):
        for c in range(padding,img.shape[1]):
            if temp[r,c]==0: continue
            if np.sum(img[r-padding:r+padding+1, c-padding:c+padding+1])<9:
                cnt[r,c]=255
    return cnt


def dilate(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def erode(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def opening(img, kd, ke, num):
    for _ in range(num):
        img = dilate(img, kd)
        img = erode(img, ke)
    return img

def closing(img, kd, ke, num):
    for _ in range(num):
        img = erode(img, ke)
        img = dilate(img, kd)
    return img

def make_border(img, padding, value=0):
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=value)
    return img
def LBP(img, size=(64,64), R=1, P=8):
    h,w=img.shape[0], img.shape[1]
    gray = bgr2gray(img)
    gray = resize(gray, size)
    # cv2show(gray, 'gray')
    padding=1
    gray = make_border(gray, padding)
    hist = [0 for _ in range(P+1)]
    for i in range(padding, gray.shape[0]-1):
        for j in range(padding, gray.shape[1]-1):
            win = gray[i-padding:i+padding+1, j-padding:j+padding+1]
            intensities = interpolate(win,R,P)
            # print(intensities)
            pattern = np.where(intensities >= gray[i,j],1,0)
            # print(pattern)
            enc = encode(pattern, P)
            # print(enc)
            hist[enc-1]+=1

    return hist



def interpolate(win,R=1,P=8):
    X = np.cos(np.array([(np.pi / (P / 2)) * p for p in range(P)])) * R
    Y = np.sin(np.array([(np.pi / (P / 2)) * p for p in range(P)])) * R
    vals = [0 for _ in range(P)]
    # print(X)
    # print(Y)
    for i in range(1,len(X),2):
        p1 = np.array((int(np.floor(1+X[i])), int(np.floor(1+Y[i]))))
        p2 = np.array((int(np.ceil(1+X[i])), int(np.floor(1+Y[i]))))
        p3 = np.array((int(np.floor(1+X[i])), int(np.ceil(1+Y[i]))))
        p4 = np.array((int(np.ceil(1+X[i])), int(np.ceil(1+Y[i]))))
        # print("points",p1,p2,p3,p4)
        p = np.array((X[i], Y[i]))
        
        d1 = 1/np.linalg.norm(p1-p)
        d2 = 1/np.linalg.norm(p2-p)
        d3 = 1/np.linalg.norm(p3-p)
        d4 = 1/np.linalg.norm(p4-p)
        # print("distances",d1,d2,d3,d4)
        inten = d1*win[p1[0],p1[1]] + d2*win[p2[0],p2[1]] + d3*win[p3[0],p3[1]] + d4*win[p4[0],p4[1]]
        inten/=(d1+d2+d3+d4+1e-16)
        vals[i] = inten
    vals[0] = win[2,1]
    vals[2] = win[1,2]
    vals[4] = win[0,1]
    vals[6] = win[1,0]
    return np.array(vals)

def encode(pattern,P):
    # print(pattern)
    pattern = pattern.tolist()
    bv = BitVector.BitVector(bitlist=pattern)
    val = [int(bv << 1) for _ in range(P)]
    minbv = BitVector.BitVector(intVal = min(val), size=P)
    runs = minbv.runs()
    if len(runs)>2:
        enc = P+1
    elif len(runs) == 1 and runs[0][0]=='1':
        enc = P
    elif len(runs) == 1 and runs[0][0]=='0':
        enc = 0
    else:
        enc = len(runs[1])
    return enc


def getLabel(name):
    label=0
    if "cloudy" in name : label = 0
    if "rain" in name : label = 1
    if "shine" in name : label = 2
    if "sunrise" in name : label = 3
    return label

def train(X_train,Y_train, X_test):
    clf = svm.SVC()
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred

def display_conf_mat(conf_mat, labels, path, acc_str):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    disp.plot()
    plt.title("Accuracy :" + acc_str)
    plt.savefig(path)
    plt.show()
def evaluate(y_true, y_pred, path, labels):
    acc_score = accuracy_score(y_true, y_pred)
    conf_matx = confusion_matrix(y_true, y_pred)
    print("Model Accuracy: ", acc_score)
    acc_str = str(acc_score*100) + " %"
    print("Confusion Matrix: ", conf_matx)
    # labels = [i for i in range(4)]
    display_conf_mat(conf_matx, labels, path, acc_str)

def classify(X_train, Y_train, X_test, Y_test, path, labels):
    pred_labels = train(X_train, Y_train, X_test)
    evaluate(Y_test, pred_labels, path, labels)

def vgg_feature_extractor(img, model_path, size=(256,256)):
    vgg = VGG19()
    vgg.load_weights(model_path)
    img = transform.resize(img, size)
    feature = vgg(img)
    return feature

def channelNorm(features):
    features = rearrange(features, 'c h w -> c (h w)')
    mu = np.mean(features, axis=1)
    var = np.var(features, axis=1)
    texture_des = np.zeros(mu.shape[0]+var.shape[0], dtype=mu.dtype)
    texture_des[::2] = mu
    texture_des[1::2] = var
    return texture_des

def cannyEdge(img, blur=False, thresh1=300, thresh2=300,kernel=3):
    gray = bgr2gray(img)
    if blur:
        gray=gauss_blur(gray, 3)
    edgeMap = cv2.Canny(gray, thresh1,thresh2,None,kernel)
    return edgeMap

def houghLines(edgeMap, thresh):
    lines = cv2.HoughLines(edgeMap,1, np.pi/180, thresh)
    pts = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        pts.append([x1, y1, x2, y2])
    return pts
def houghLinesP(edgeMap):
    lines = cv2.HoughLinesP(edgeMap, 1, np.pi/180, 15, None, 15, 15)
    return lines

def calcAngle(m1,m2):
    return np.arctan((m1-m2)/(1+(m1*m2)))
def filterLines(lines):
    l1 = lines[0]
    l1_hc = make_line_hc(np.array((l1[0], l1[1])), np.array((l1[2], l1[3])))
    m1 = -l1_hc[0] / (l1_hc[1] + 1e-6)
    angle_x_axis = calcAngle(m1, 0)
    if (angle_x_axis**2) <= (np.pi/4)**2:
        type = "h"
    elif (angle_x_axis ** 2) > (np.pi / 4) ** 2:
        type = "v"

    type1 = []
    type1.append(l1)
    type2 = []
    for i in range(1,len(lines)):
        l = lines[i]
        l_hc = make_line_hc(np.array((l[0], l[1])), np.array((l[2], l[3])))
        m = -l_hc[0] / (l_hc[1] + 1e-6)
        angle = calcAngle(m,m1)
        if (angle**2) <= (np.pi/4)**2:
            type1.append(l)
        elif (angle**2) > (np.pi/4)**2:
            type2.append(l)
    # print(len(type1), len(type2), type)
    return type1, type2, type


def clubLines(lines, img):
    print(len(lines))
    h,w,c=img.shape
    polygon = Polygon([(0, 0), (0, w), (h, w), (h, 0)])
    idx = [[] for _ in range(len(lines))]
    idxs = []
    for i in range(len(lines)-1):
        l1 = lines[i]
        l1_hc = make_line_hc(np.array((l1[0], l1[1])), np.array((l1[2], l1[3])))
        for j in range(i+1,len(lines)):
            l2 = lines[j]
            l2_hc = make_line_hc(np.array((l2[0], l2[1])), np.array((l2[2], l2[3])))
            pt = getIntersectionPoint(l1_hc, l2_hc)
            point = Point(pt)
            if polygon.contains(point):
                print(pt)
                idxs.append(j)

    refined_lines =[]
    for i in range(len(lines)):
        if i not in idxs:
            refined_lines.append(lines[i])
    print(refined_lines)
    print(len(refined_lines))
    return refined_lines

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
def refineLines(lines, type, lines_count):
    x_hc = np.array((0,1,0))
    y_hc = np.array((-1,0,0))
    intersections = []
    data = []
    for i in range(len(lines)):
        l=lines[i]
        l_hc = make_line_hc(np.array((l[0], l[1])), np.array((l[2], l[3])))
        if type=="v":
            pt = getIntersectionPoint(l_hc, x_hc)
            data.append(pt[0])
        elif type=="h":
            pt = getIntersectionPoint(l_hc, y_hc)
            data.append(pt[1])
    data=np.array(data)
    data = data.reshape(-1,1)
    clusters = KMeans(n_clusters=lines_count)
    clusters.fit(data)
    cluster_centers = clusters.cluster_centers_
    cluster_idxs = clusters.predict(data)
    cluster_lines = [[] for _ in range(lines_count)]
    for i in range(len(cluster_idxs)):
        cluster_lines[cluster_idxs[i]].append(lines[i])

    refined_lines=[]
    for cluster_line in cluster_lines:
        refined_lines.append(cluster_line[0])

    for refined_line in refined_lines:
        refined_line_hc = make_line_hc(np.array((refined_line[0], refined_line[1])), np.array((refined_line[2], refined_line[3])))
        if type=="v":
            pt = getIntersectionPoint(refined_line_hc, x_hc)
            intersections.append((refined_line, pt[0]))
        elif type=="h":
            pt = getIntersectionPoint(refined_line_hc, y_hc)
            intersections.append((refined_line, pt[1]))

    sorted_refined_lines = np.array(sorted(intersections, key=lambda x:x[1]))[:,0].tolist()
    return sorted_refined_lines


def getCorners(vlines, hlines):
    corners = []
    for hl in hlines:
        hl_hc = make_line_hc(np.array((hl[0], hl[1])), np.array((hl[2], hl[3])))
        for vl in vlines:
            vl_hc = make_line_hc(np.array((vl[0], vl[1])), np.array((vl[2], vl[3])))
            pt = getIntersectionPoint(hl_hc, vl_hc)
            corners.append(pt)

    corners=np.array(corners)
    return corners

def sortLines(lines, type):
    if type=='v':
        lines = sorted(lines, key=lambda x:x[0])
    if type=='h':
        lines = sorted(lines, key=lambda x:x[1])

    return lines
def plotLinesP(lines, img):
    for i in range(len(lines)):
        l = lines[i][0]
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)
    return img

def plotLines(lines, img, color):
    plot_img = np.copy(img)
    for i in range(len(lines)):
        l = lines[i]
        cv2.line(plot_img, (l[0], l[1]), (l[2], l[3]), color, 2, cv2.LINE_AA)
    return plot_img


def cv2HarrisCorner(img):
    gray = bgr2gray(img)
    gray = cv2.cornerHarris(gray,2,3,0.04)
    gray = cv2.dilate(gray, None)
    corners = np.argwhere(gray > 0.01 * gray.max())
    img[gray > 0.01 * gray.max()] = [0, 0, 255]
    return corners, img
def getIntersectionPoint(l1, l2):
    pts = np.cross(l1, l2)
    pts = pts/(pts[2]+1e-6)
    return np.array([int(pts[0]), int(pts[1])])

def plotPoints(pts, img, mode,color):
    plot_img = np.copy(img)
    count=1
    if mode == "harris":
        for pt in pts:
            cv2.circle(plot_img, (int(pt[1]), int(pt[0])), radius=2, color=color, thickness=-1)
            cv2.putText(plot_img, str(count), (int(pt[1]), int(pt[0])), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1, cv2.LINE_AA)
            count+=1
    else:
        for pt in pts:
            cv2.circle(plot_img, (int(pt[0]), int(pt[1])), radius=2, color=color, thickness=-1)
            cv2.putText(plot_img, str(count), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1, cv2.LINE_AA)
            count += 1

    return plot_img

def Vij(H,i,j):
    v=np.zeros(6)
    v[0]=H[0,i]*H[0,j]
    v[1]=H[0,i]*H[1,j]+H[1,i]*H[0,j]
    v[2]=H[1,i]*H[1,j]
    v[3]=H[2,i]*H[0,j]+H[0,i]*H[2,j]
    v[4]=H[2,i]*H[1,j]+H[1,i]*H[2,j]
    v[5]=H[2,i]*H[2,j]
    return(v)

def V(H):
    v12=Vij(H,0,1)
    v11=Vij(H,0,0)
    v22=Vij(H,1,1)
    return np.array(v12), np.array((v11-v22))
def omegaCalc(H):
    Vmat = np.ones((len(H) * 2, 6))
    for i in range(len(H)):
        v1, v2 = V(H[i])
        Vmat[2 * i] = v1
        Vmat[(2*i)+1] = v2

    u,d,ut = np.linalg.svd(Vmat)
    b = ut[-1]
    w = [[b[0],b[1],b[3]],[b[1],b[2],b[4]],[b[3],b[4],b[5]]]
    w = np.array(w)
    return w

def zhangK(w):
    y0 = ((w[0,1]*w[0,2])-(w[0,0]*w[1,2]))/((w[0,0]*w[1,1])-(w[0,1]**2))
    A = w[2,2] - ((w[0,2]**2 + (y0*((w[0,1]*w[0,2])-(w[0,0]*w[1,2]))))/(w[0,0]))
    ax = np.sqrt(A/w[0,0])
    ay = np.sqrt((A*w[0,0])/((w[0,0]*w[1,1])-(w[0,1]**2)))
    s = -(w[0,1]*(ax**2)*ay)/(A)
    x0 = ((s*y0)/ay) - ((w[0,2]*(ax**2))/A)
    K = np.array([[ax, s, x0],[0, ay, y0],[0,0,1]])
    return K

def zhangRT(H, K):
    R = []
    T = []
    Kinv = np.linalg.inv(K)
    for h in H:
        r12t = np.dot(Kinv, h)
        norm = 1/np.linalg.norm(r12t[:,0])
        r12t = norm * r12t
        r3 = np.cross(r12t[:,0], r12t[:,1])
        r = np.column_stack((r12t[:,:2], r3))
        u,d,ut = np.linalg.svd(r)
        r = np.dot(u,ut)
        R.append(r)
        T.append(r12t[:,2])
    return R,T

def paramComb(R, T, K):
    _W = []
    comb_array = []
    K_arr = [K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]]
    comb_array.append(K_arr)
    for i in range(len(R)):
        r = R[i]
        t = np.array(T[i])
        angle = np.arccos((np.trace(r) - 1)/2)
        _w = (angle/(2*np.sin(angle))) * np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
        comb_array.append(_w)
        comb_array.append(t)
    comb_array = np.array(comb_array)
    comb_array = np.concatenate(comb_array)
    return comb_array

def paramSep(params, N, rd=False):
    if rd:
        k1,k2 = params[0:2]
        params = params[2:]
    k = params[:5]
    k = params[:5]
    K = np.array([[k[0],k[1],k[2]],[0,k[3],k[4]],[0,0,1]])
    rem_params = params[5:]
    rem_params = rearrange(rem_params, '(c h) -> c h', c=N, h=6)
    # print(rem_params.shape)
    R=[]
    T=[]
    for i in range(N):
        _w = rem_params[i][:3]
        t = rem_params[i][3:]
        norm = np.linalg.norm(_w)
        _W = np.array([[0, -_w[2], _w[1]], [_w[2], 0, -_w[0]], [-_w[1], _w[0], 0]])
        r = np.eye(3) + (np.sin(norm) / norm) * _W + ((1 - np.cos(norm)) / (norm ** 2)) * np.dot(_W,_W)
        T.append(t)
        R.append(r)
    if rd:
        return R,T,K,k1,k2
    return R,T,K

def CameraCalibrationHomography(K, R, T):
    Hcam = []
    for i in range(len(R)):
        RT = np.column_stack((R[i][:,:2], T[i].T))
        hcam = np.dot(K, RT)
        Hcam.append(hcam)

    return Hcam

def CameraReporjection(Hcam, CP_Corners):
    reprojCorners = []
    for i in range(len(Hcam)):
        hcam = np.array(Hcam[i])
        corners = CP_Corners
        proj_corners = []
        for corner in corners:
            corner_hc = cvrt2homo(corner)
            # print(corner_hc)
            proj_corner = np.dot(hcam, corner_hc.T)
            # print(proj_corner)
            proj_corner = proj_corner/(proj_corner[2]+1e-6)
            proj_corners.append((int(proj_corner[0]), int(proj_corner[1])))
        reprojCorners.append(proj_corners)
    return reprojCorners

def CameraReporjection2BaseImg(Hcam, CP_Corners):
    reprojCorners = []
    for i in range(len(Hcam)):
        hcam = np.array(Hcam[i])
        corners = CP_Corners[i]
        proj_corners = []
        for corner in corners:
            corner_hc = cvrt2homo(corner)
            # print(corner_hc)
            proj_corner = np.dot(hcam, corner_hc.T)
            # print(proj_corner)
            proj_corner = proj_corner/(proj_corner[2]+1e-6)
            proj_corners.append((int(proj_corner[0]), int(proj_corner[1])))
        reprojCorners.append(proj_corners)
    return reprojCorners

def costFunCameraCaleb(params, CP_Corners, Corners, rd=False):
    if rd:
        R,T,K = paramSep(params[2:], len(Corners))
        k1,k2 = params[0:2]
        x0 = params[4]
        y0 = params[6]
    else:
        R,T,K = paramSep(params, len(Corners))
    RP_Corners = []
    for i in range(len(R)):
        RT = np.column_stack((R[i][:,:2], T[i].T))
        hcam = np.dot(K, RT)
        proj_corners = []
        for corner in CP_Corners:
            corner_hc = cvrt2homo(corner)
            proj_corner = np.dot(hcam, corner_hc.T)
            proj_corner = proj_corner / (proj_corner[2] + 1e-6)
            proj_corners.append((proj_corner[0],proj_corner[1]))
        RP_Corners.append(proj_corners)

    if rd:
        RP_Corners = radialDistort(RP_Corners, k1,k2,x0,y0)
    X = rearrange(np.array(RP_Corners), 'b c h -> (b c h)')
    f = rearrange(np.array(Corners), 'b c h -> (b c h)')
    error = (X-f)
    return error

def radialDistort(Corners, k1,k2,x0,y0):
    RP_Corners=[]
    # print(np.array(Corners).shape)
    for i in range(len(Corners)):
        corner = np.array(Corners[i])
        x = corner[:,0]
        y = corner[:,1]
        r = (x-x0)**2 + (y-y0)**2
        _x = x + ((x-x0)*(k1*(r**2) + k2*(r**4)))
        _y = y + ((y-y0)*(k1*(r**2) + k2*(r**4)))
        RP_Corners.append(np.column_stack((_x,_y)))
    return RP_Corners


def getError(diff):
    N = len(diff)//2
    diff = rearrange(diff, '(c h) -> c h', c=N, h=2)
    norm = np.linalg.norm(diff, axis=1)
    max_diff = np.max(norm)
    mean = np.mean(norm)
    var = np.var(norm)
    return max_diff, mean, var


def reprojectCorners(Hcam, cp_id, Corners):
    h_base = Hcam[cp_id]
    reproj_H = []
    for i in range(len(Hcam)):
        h = Hcam[i]
        reproj_h = np.dot(h_base, np.linalg.inv(h))
        reproj_H.append(reproj_h)
    reproj_corners = CameraReporjection2BaseImg(reproj_H, Corners)
    return reproj_corners



def reprojError(cp_corners, Corners):
    cp_corners = np.array(cp_corners)
    Corners = np.array(Corners)
    cp_corners = repeat(cp_corners, 'h w -> c h w', c=len(Corners))
    X = rearrange(cp_corners, 'b c h -> (b c h)')
    f = rearrange(Corners, 'b c h -> (b c h)')
    error = (X-f)
    return error


def normPts(pts):
    mean_pts = np.mean(pts, axis=0)
    d=pts-mean_pts
    d**=2
    d=np.mean(np.sqrt(np.sum(d,axis=1)))
    c=np.sqrt(2)/d
    T=np.array([[c,0,-c*mean_pts[0]],[0,c,-c*mean_pts[1]],[0,0,1]])
    pts_hcs = pts2hc(pts)
    pts_hcs_norm = np.dot(T, pts_hcs.T)
    pts_hcs_norm = rearrange(pts_hcs_norm,'c h -> h c')
    return pts_hcs_norm[:,:2], T

def calc_F(pts1, pts2, T1, T2):
    A = np.ones((len(pts1), 9))
    for i in range(len(pts1)):
        A[i] = np.array([pts2[i][0]*pts1[i][0], pts2[i][0]*pts1[i][1], pts2[i][0],
                         pts2[i][1]*pts1[i][0], pts2[i][1]*pts1[i][1], pts2[i][1], pts1[i][0], pts1[i][1], 1])

    U,D,UT=np.linalg.svd(A)
    uF = UT[-1]
    uF = rearrange(uF, '(c h)-> c h', c=3, h=3)
    cF = cond_F(uF)
    cF = normF(cF, T1, T2)
    return cF

def cond_F(uF):
    V, D, VT = np.linalg.svd(uF)
    D[2] = 0
    D = np.diag(D)
    cF = np.dot(np.dot(V, D), VT)
    return cF

def normF(cF, T1, T2):
    cF = np.dot(np.dot(T2.T, cF), T1)
    cF = cF / cF[2, 2]
    return cF


def findEpipole(F):
    U,D,V=np.linalg.svd(F)
    eL = np.transpose(V[-1,:])
    eR = U[:,-1]
    eL = eL/eL[2]
    eR = eR/eR[2]
    Ex = np.array([[0, -eR[2],eR[1]],[eR[2],0,-eR[0]],[-eR[1],eR[0],0]])
    return eL,eR,Ex

def getP(F,e,Ex):
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P2 = np.column_stack((Ex@F, e.T))
    return P1, P2

def FCostFun(f, pts1, pts2):
    F = rearrange(f,'(c h)-> c h ',c=3,h=3)
    eL,eR,Ex = findEpipole(F)
    P1,P2=getP(F,eR,Ex)
    pts1_hc = pts2hc(pts1)
    pts2_hc = pts2hc(pts2)
    error=[]
    for i in range(len(pts1)):
        A = np.array([pts1_hc[i][0]*P1[2,:]-P1[0,:],pts1_hc[i][1]*P1[2,:]-P1[1,:],pts2_hc[i][0]*P2[2,:]-P2[0,:],pts2_hc[i][1]*P2[2,:]-P2[1,:]])
        U,D,UT = np.linalg.svd(A)
        X = UT[-1,:]
        X = X/X[-1]
        pts1_est = P1@X
        pts2_est = P2@X
        pts1_est=pts1_est/pts1_est[-1]
        pts2_est=pts2_est/pts2_est[-1]
        error.append(np.linalg.norm(pts1_est-pts1_hc[i])**2)
        error.append(np.linalg.norm(pts2_est-pts2_hc[i])**2)
    return np.array(error)


def get_angle(e,h,w):
    return np.arctan2(e[1] - h / 2, -(e[0] - w / 2))

def get_R(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

def get_f(e, angle, w, h):
    return (e[0] - w / 2) * np.cos(angle) - (e[1] - h / 2) * np.sin(angle)

def get_G(f):
    return np.array([[1, 0, 0], [0, 1, 0], [-1 / f, 0, 1]])

def get_T(w,h,img_center, offset=False):
    if offset:
        return np.array([[1, 0, w/2 - img_center[0]], [0, 1, h/2 - img_center[1]], [0, 0, 1]])
    return np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])

def get_TGRT(T1,G,R,T2):
    return np.dot(np.dot(T1,G), np.dot(R,T2))


def get_Ha(pts1, pts2, H1, H2):
    pts1_hc = pts2hc(pts1)
    pts2_hc = pts2hc(pts2)

    x1 = (H1@(pts1_hc.T)).T
    x2 = (H2@(pts2_hc.T)).T

    x1[:, 0] /= x1[:, 2]
    x1[:, 1] /= x1[:, 2]
    x1[:, 2] /= x1[:, 2]
    x2[:, 0] /= x2[:, 2]

    h = np.linalg.pinv(x1)@x2[:, 0]
    H = np.array([[h[0], h[1], h[2]], [0, 1, 0], [0, 0, 1]])
    return H
def get_homographies(height, width, pts1, pts2,e1,e2):
    I = np.eye(3).astype(np.uint8)
    T = get_T(width,height,[0,0])
    angle = get_angle(e2,height,width)
    R = get_R(angle)
    f = get_f(e2,angle,width,height)
    G= get_G(f)
    H2c = get_TGRT(I,G,R,T)

    img2C = H2c@np.array([width/2, height/2, 1])
    img2C/=img2C[2]

    T2 = get_T(width, height, [img2C[0],img2C[1]], True)
    H2 = get_TGRT(T2,G,R,T)
    H2/=H2[2,2]

    angle = get_angle(e1, height, width)
    R = get_R(angle)
    f = get_f(e1, angle, width, height)
    G = get_G(f)
    H0 = get_TGRT(T,G,R,T)

    Ha = get_Ha(pts1, pts2, H0, H2)

    H1c = np.dot(Ha,H0)

    img1C = H1c @ np.array([width / 2, height / 2, 1])
    img1C /= img1C[2]

    T1 = get_T(width, height, [img1C[0],img1C[1]], True)

    H1 = T1@H1c
    H1 /= H1[2,2]
    return H1,H2


def applyHomography(img, H):
    h,w,c=img.shape
    wpa,hpa,wpe,hpe=findmaxmin(h,w,H)
    wt=wpa-wpe
    ht=hpa-hpe
    print(wpa,hpa,wpe,hpe)
    hinv=np.linalg.pinv(H)
    targetImg = np.zeros((ht,wt,3),dtype='uint8')
    for c in tqdm(range(int(wt))):
        for r in range(int(ht)):
            X_prime=xpeqhx(c,r,hinv)
            if X_prime[1]<=0:
                X_prime[1]+=hpe
            if X_prime[0] <= 0:
                X_prime[0]+=wpe
            if X_prime[1] < h and X_prime[0] < w and X_prime[0]>=0 and X_prime[1]>=0:
                targetImg[r][c] = img[X_prime[1]][X_prime[0]]
    return targetImg



def filterDim(img):
    img_gray = bgr2gray(img)
    R,C = np.where(img_gray>0)
    minH = np.min(R)
    maxH = np.max(R)
    minW = np.min(C)
    maxW = np.max(C)
    return minH, maxH, minW, maxW

def filterImg(img, minH, maxH, minW, maxW):
    return img[minH:maxH, minW:maxW]

def epiLine(F,pts_hc,img):
    eplineImg = img.copy()
    lines = np.dot(F,pts_hc.T)
    lines=lines.T
    for line in lines:
        cv2.line(eplineImg, (0,int(-line[2]/line[1])), (img.shape[1]-1, int(-(line[2]+line[0]*(img.shape[1]-1))/line[1])), pick_color(), 2)
    return eplineImg


def plot_pts(img, pts):
    img_pts = img.copy()
    color=pick_color()
    for pt in pts:
        cv2.circle(img_pts, (pt[0],pt[1]), radius=5, color=color, thickness=-1)

    return img_pts


# def get_corrs(edge1, edge2, r_win, col_win):
def hc2pts(pts_hc):
    pts_hc = pts_hc.T
    pts = pts_hc[:,:-1]
    return pts

def get_corrs_sec_img(edge, row, col, col_k):
    win = edge[row, col:col+col_k+1]
    col_idxs = np.where(win>0)[0]
    if len(col_idxs)>0:
        col_idxs += col
        return col_idxs
    return np.empty(0)
def get_corrs(edge1, edge2, col_k):
    corrs=[]
    for r in range(edge1.shape[0]):
        c_L = np.where(edge1[r]>0)[0]
        if len(c_L)>0:
            for c in c_L:
                c_R = get_corrs_sec_img(edge2, r, c, col_k)
                if len(c_R)>0:
                    edge2[r, c_R]=0
                    corrs.append([[c,r],[c_R[0],r]])

    return np.array(corrs)

def ssd_corrs(pts1, pts2, img1, img2, win_size):
    combined_pt_set=[]
    patches1=[]
    patches2=[]
    pad=int((win_size-1)/2)
    img1=cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    img2=cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for pt in pts1:
        patches1.append(extarct_patch(pt, win_size, img1))
    for pt in pts2:
        patches2.append(extarct_patch(pt, win_size, img2))
    patches1=np.array(patches1)
    patches2=np.array(patches2)
    dist_vec = np.linalg.norm((patches1-patches2)**2, axis=1)
    sorted_indexes = np.argsort(dist_vec)
    return sorted_indexes

def plot_corrs(img1, img2, corres, idxs, max_corrs):
    img1Height = img1.shape[0]
    img2Height = img2.shape[0]
    if (img1Height<img2Height):
        img1 = np.concatenate((img1, np.zeros((img2Height-img1Height, img1.shape[1],3), np.uint8)),0)
    elif (img2Height<img1Height):
        img2 = np.concatenate((img2, np.zeros((img1Height-img2Height, img2.shape[1],3), np.uint8)),0)
    newImg = np.concatenate((img1, img2),1)
    img2OffsetX = img1.shape[1]
    for i in range(max_corrs):
        ptSet = corres[idxs[i]]
        pt1 = ptSet[0]
        pt2 = np.array(ptSet[1])+[img2OffsetX,0]
        color = pick_color()
        cv2.line(newImg, pt1, pt2, color, 2, cv2.LINE_AA)
        cv2.circle(newImg, pt1,4,color, -1)
        cv2.circle(newImg, pt2,4,color, -1)
    return newImg

def Triangulate(pts1, pts2, P1, P2):
    world_pts=[]
    for i in range(len(pts1)):
        A = np.array([pts1[i][0] * P1[2, :] - P1[0, :],
                      pts1[i][1] * P1[2, :] - P1[1, :],
                      pts2[i][0] * P2[2, :] - P2[0, :],
                      pts2[i][1] * P2[2, :] - P2[1, :]])
        U, D, UT = np.linalg.svd(A)
        X = UT[-1, :].T
        X = X / X[-1]
        world_pts.append(X)
    return np.asarray(world_pts)


def plot_3D_point_cloud(world_pts, path, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(world_pts[:, 0], world_pts[:, 1], world_pts[:, 2])
    fig_name = os.path.join(path, name)
    plt.savefig(fig_name)
    plt.show()

def plot_3D_projection(world_pts, path, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pairs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 1],
             [1, 5],
             [4, 6],
             [5, 7],
             [6, 8],
             [8, 7]]
    for pair in pairs:
        ax.plot([world_pts[pair[0]][0], world_pts[pair[1]][0]],
                [world_pts[pair[0]][1], world_pts[pair[1]][1]],
                [world_pts[pair[0]][2], world_pts[pair[1]][2]])
    fig_name = os.path.join(path,name)
    plt.savefig(fig_name)
    plt.show()

def getRefinedCorrs(H, pts):
    ref_pts = []
    for pt in pts:
        [newX, newY, newZ] = np.dot(H, [pt[0], pt[1], 1.0])
        ref_pts.append([round(newX/newZ), round(newY/newZ)])
    return np.array(ref_pts)



def apply_census_r2l(img1, img2, d_max, win_size=3):
    pad = win_size // 2
    bounds = d_max+pad
    d_map = np.zeros(img1.shape, dtype='uint8')
    img1 = cv2.copyMakeBorder(img1, bounds, bounds, bounds, bounds, cv2.BORDER_CONSTANT, value=255)
    img2 = cv2.copyMakeBorder(img2, bounds, bounds, bounds, bounds, cv2.BORDER_CONSTANT, value=255)
    h,w = img1.shape
    for r in trange(bounds, h-bounds):
        for c in range(w-bounds-1, bounds-1,-1):
            cost = []
            p1 = get_patch(img1, r, c,pad)
            b1 = np.ravel((p1 > p1[pad,pad])*1)
            for d in range(d_max+1):
                p2 = get_patch(img2,r,c-d,pad)
                b2 = np.ravel((p2>p2[pad,pad])*1)
                cost.append(sum(b1^b2))
            d_map[r-bounds,c-bounds] = np.argmin(cost)

    return d_map
def apply_census_l2r(img1, img2, d_max, win_size=3):
    pad = win_size // 2
    bounds = d_max+pad
    d_map = np.zeros(img1.shape, dtype='uint8')
    img1 = cv2.copyMakeBorder(img1, bounds, bounds, bounds, bounds, cv2.BORDER_CONSTANT, value=255)
    img2 = cv2.copyMakeBorder(img2, bounds, bounds, bounds, bounds, cv2.BORDER_CONSTANT, value=255)
    h,w = img1.shape
    for r in trange(bounds, h-bounds):
        for c in range(bounds, w-bounds-1):
            cost = []
            p1 = get_patch(img1, r, c,pad)
            b1 = np.ravel((p1 > p1[pad,pad])*1)
            for d in range(d_max+1):
                p2 = get_patch(img2,r,c+d,pad)
                b2 = np.ravel((p2>p2[pad,pad])*1)
                cost.append(sum(b1^b2))
            d_map[r-bounds,c-bounds] = np.argmin(cost)

    return d_map
def get_dmax(disp):
    disp = bgr2gray(disp)
    disp = disp.astype(np.float32) / 4.0
    disp = disp.astype(np.uint8)
    return disp,np.max(disp)

def view_dmap(dmap, name, path):
    dmap = (dmap/np.max(dmap)*255).astype(np.uint8)
    save_img_v2(name, path, dmap)
    # cv2show(dmap,'dmap')

def error_disp(dmap, gt_dmap,name,path, delta=2):
    error = abs(dmap - gt_dmap)
    error = ((error<=delta)*255).astype(np.uint8)
    save_img_v2(name, path, error)
    # cv2show(error,'error')
    de = cv2.countNonZero(error)
    dg = cv2.countNonZero(gt_dmap)
    print(de,dg)
    return de/dg


# def data_loader(path):
#     img_paths = glob.glob(path+'/*.png')
#     X=[]
#     Y=[]
#     for path in img_paths:
#         Y.append(int(path.split('.')[0].split('/')[-1].split('_')[0]))
#         img = readImgCV(path)
#         gray_img = bgr2gray(img)
#         gray_img = rearrange(gray_img, 'h w -> (h w)')
#         X.append(gray_img)
#     # print(len(X), len(Y))
#     return np.array(X), np.array(Y)
#
# def normalizeData(X):
#     return X/np.linalg.norm(X)
#
# def prep_data(X):
#     X = normalizeData(X)
#     m = np.mean(X, axis=1)
#     m = rearrange(m, '(c h) -> c h', h=1)
#     X = X - m
#     return X, m
# def PCA(X):
#     X,m=prep_data(X)
#     C = X@X.T
#     w,v = np.linalg.eig(C)
#     sort_idx = np.argsort(w)
#     v = v[~sort_idx]
#     W = X.T@v
#     W = W/np.linalg.norm(W)
#     return m, W.T
#
# def nn(y_test, y_train, Y_train):
#     Y_pred=[]
#     for i in range(y_test.shape[0]):
#         test_feature = y_test[i,:]
#         diff = np.linalg.norm(y_train-test_feature, axis=1)
#         idx = np.argmin(diff)
#         Y_pred.append(Y_train[idx])
#     Y_pred = np.array(Y_pred)
#     match_num = len(np.where(Y_pred==Y_train)[0])
#     return match_num

def data_loader(path):
    img_paths = glob.glob(path+'/*.png')
    X=[]
    Y=[]
    for path in img_paths:
        Y.append(int(path.split('.')[0].split('/')[-1].split('_')[0]))
        img = readImgCV(path)
        gray_img = bgr2gray(img)
        gray_img = rearrange(gray_img, 'h w -> (h w)')
        X.append(gray_img)
    X = np.array(X)
    Y=np.array(Y)
    X=X.T
    X=X/np.linalg.norm(X)
    m = np.mean(X,axis=1)
    m = rearrange(m, '(c h) -> c h', h=1)
    # print(X.shape,Y.shape,m.shape)
    return X,Y,m



def prep_data(X):
    # X = normalizeData(X)
    m = np.mean(X, axis=1)
    m = rearrange(m, '(c h) -> c h', h=1)
    X = X - m
    return X, m
def PCA(X,m):
    # X,m=prep_data(X)
    X=X-m
    C = X.T@X
    # print(C.shape)
    w,v = np.linalg.eig(C)
    sort_idx = np.argsort(w)
    v = v[~sort_idx]
    W = X@v
    W = W/np.linalg.norm(W)
    # print(W.shape)
    return W

def NearestNeighbor(y_test, y_train, Y_train, Y_test):
    # print(y_train.shape)
    Y_pred=[]
    for i in range(y_test.shape[0]):
        test_feature = y_test[i,:]
        # print(test_feature.shape)
        diff = np.linalg.norm(y_train-test_feature, axis=1)
        # print(diff.shape)
        idx = np.argmin(diff)
        # print(idx)
        Y_pred.append(Y_train[idx])
    Y_pred = np.array(Y_pred)
    # print(Y_pred)
    # print(Y_train)
    match_num = len(np.where(Y_pred==Y_test)[0])
    return match_num

def idx_per_class(Y, num_classes):
    idx_array=[]
    for i in range(num_classes):
        idxs = (np.where(Y==i+1)[0])
        idx_array.append(idxs)
    return np.array(idx_array)

def LDA(X,m, idx_array, num_classes):
    M = []
    for idx in idx_array:
        mi = np.mean(X[:,idx], axis=1)
        mi = rearrange(mi, '(c h) -> c h', h=1)
        M.append(mi-m)
    M = np.array(M)
    M = rearrange(M, 'c h w -> c (h w)')
    _mtm = M@M.T
    _mtm_eig_val, _mtm_eig_vec = np.linalg.eig(_mtm)
    sort_idx = np.argsort(_mtm_eig_val)
    _mtm_eig_val = _mtm_eig_val[~sort_idx]
    _mtm_eig_vec = _mtm_eig_vec[~sort_idx]
    g_idx = np.where(_mtm_eig_val>1e-5)[0]
    _mtm_eig_val = _mtm_eig_val[g_idx]
    _mtm_eig_vec = _mtm_eig_vec[g_idx]
    V = M.T@_mtm_eig_vec.T
    Db = np.eye(len(g_idx))*(_mtm_eig_val**(-0.5))
    Z  =V@Db
    Xn = Z.T@X
    XntXn = Xn@Xn.T
    val,vec = np.linalg.eig(XntXn)
    sorted_idx = np.argsort(val)
    vec = vec[~sorted_idx]
    W = Z@vec
    W=W/np.linalg.norm(W)
    return W

def plot_acc(pca_list, lda_list, K ):
    plt.plot(K, pca_list,'ro')
    plt.plot(K, pca_list, label='PCA')
    plt.plot(K, lda_list,'bo')
    plt.plot(K, lda_list, label='LDA')
    plt.legend()
    plt.xlabel('Dimensions')
    plt.ylabel('Accuracy')
    plt.show()


def get_feature(img, max_size=2):
    feature=[]
    gx, gy = Sobel()
    dx = convolve2d(img, gx, mode="same")
    dy = convolve2d(img, gy, mode="same")
    dx = rearrange(dx, 'c h -> (c h)')
    dy = rearrange(dy, 'c h -> (c h)')
    feature.append(dx)
    feature.append(dy)
    for size in range(1, max_size+1):
        gx, gy = Haar_Wavelet(size)
        dx = convolve2d(img, gx, mode="same")
        dy = convolve2d(img, gy, mode="same")
        dx = rearrange(dx, 'c h -> (c h)')
        dy = rearrange(dy, 'c h -> (c h)')
        feature.append(dx)
        feature.append(dy)
    feature=np.array(feature)
    feature = rearrange(feature,'c h -> (c h)')
    # print(feature.shape)
    return feature

def get_cumsum(w,fl,idx):
    f = np.zeros((fl))
    f[idx]=w
    f_cum = np.cumsum(f)
    return f_cum

def sort_vecs(features, labels, weights):
    sort_idx = np.argsort(features)
    features=features[sort_idx]
    labels=labels[sort_idx]
    weights=weights[sort_idx]
    return features, labels, weights

def _Beta_(err):
    return err/abs(1-err+1e-6)

def _Alpha_(beta):
    return np.log(1/beta)

def update_weights(weights, beta, cls, labels):
    return weights*(beta**(1-(cls!=labels)*1))

def classifier(features, labels, weights):
    Tp = np.sum(weights[labels==1])
    Tn = np.sum(weights[labels==0])
    MIN_ERR = []
    MODEL=[]
    PRED=[]
    for i in range(features.shape[1]):
        feature = features[:,i]
        sorted_feature, sorted_labels, sorted_weights = sort_vecs(feature, labels, weights)

        pos_idx = np.where(sorted_labels==1)[0]
        neg_idx = np.where(sorted_labels==0)[0]

        Sp = sorted_weights[pos_idx]
        Sn = sorted_weights[neg_idx]

        cum_pos_weights = get_cumsum(Sp, features.shape[0], pos_idx)
        cum_neg_weights = get_cumsum(Sn, features.shape[0], neg_idx)

        err1 = cum_pos_weights+Tn-cum_neg_weights
        err2 = cum_neg_weights+Tp-cum_pos_weights
        err = np.vstack((err1,err2)).T

        min_idx = np.unravel_index(err.argmin(), err.shape)
        min_err = np.min(err)
        MIN_ERR.append(min_err)
        MODEL.append([sorted_feature[min_idx[0]],((min_idx[1]==0)*2)-1])
        if min_idx[1]==0: PRED.append((feature>=sorted_feature[min_idx[0]])*1)
        else: PRED.append((feature<sorted_feature[min_idx[0]])*1)

    MIN_ERR=np.array(MIN_ERR);MODEL = np.array(MODEL);PRED = np.array(PRED)
    err_sort_idx = np.argsort(MIN_ERR)
    MIN_ERR = MIN_ERR[err_sort_idx];MODEL = MODEL[err_sort_idx];PRED=PRED[err_sort_idx]
    return MIN_ERR[0],MODEL[0], PRED[0]

def stage(features, labels, N, weights, num_pos_samples, num_neg_samples):
    weak_classifiers=[]
    strong_classifer=np.zeros((features.shape[0]))
    thresh=0
    for _ in range(N):
        weights = weights / np.sum(weights)
        cls = classifier(features, labels, weights)
        beta = _Beta_(cls[0])
        alpha = _Alpha_(beta)
        weak_classifiers.append([cls,alpha])
        weights = update_weights(weights, beta, cls[2], labels)
        strong_classifer, thresh = update_strong_classifier(strong_classifer, alpha, thresh,cls[2])
        fp,fn=find_fp_fn(strong_classifer, num_pos_samples, num_neg_samples)
        if fp<=0.5 and fn<=0: break

    weak_classifiers = np.array(weak_classifiers)
    print(weak_classifiers, weak_classifiers.shape)

    revised_feats = features[:num_pos_samples,:]
    negFeatures = features[num_pos_samples:,:]
    negFeature_mask = cls[2][num_pos_samples:]
    negFeature_masked= negFeatures[np.where(negFeature_mask==1),:][0]
    negFeature_masked=np.reshape(negFeature_masked, (len(negFeature_masked), np.size(features,1)))
    revised_feats = np.concatenate((revised_feats,negFeature_masked),0)
    revisedLabels = np.concatenate((np.ones((num_pos_samples,1), np.uint8), np.zeros((len(negFeature_masked),1), np.uint8)),0)
    perfRates = [fp,fn]
    return revised_feats, revisedLabels, perfRates, weak_classifiers




def update_strong_classifier(str_cls, alpha, thresh, cls):
    str_cls = str_cls + (alpha*cls)
    thresh = alpha*thresh
    return (str_cls>=thresh)*1, thresh

def find_fp_fn(str_cls, num_pos_samples, num_neg_samples):
    fp = np.sum(str_cls[num_pos_samples:]==1)/num_neg_samples
    fn = 1-np.sum(str_cls[:num_pos_samples]==1)/num_pos_samples
    return fp, fn












