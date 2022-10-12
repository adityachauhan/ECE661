import os
<<<<<<< HEAD
import random
=======
from scipy.optimize import least_squares
>>>>>>> 5cced4643545f1ee75fa6a7378bc0f00ae827aab

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from einops import rearrange
from tqdm import trange, tqdm
from scipy.signal import convolve2d
import sys

def readImgCV(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cvrt2homo(pt):
    return np.append(pt, 1)



def str2np(s):
    x_prime_pts = s.split(',')
    x_prime_pts = [int(val) for val in x_prime_pts]
    x_prime_pts = np.array(x_prime_pts)
    x_prime_pts = rearrange(x_prime_pts, '(c h)-> c h ', c=4, h=2)
    return x_prime_pts

def save_img(name, path, img):
    img_path = os.path.join(path, name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    return pts_set, des


def flann_matching(pts1,pts2,des1,des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchingPts1=[]
    matchingPts2=[]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.25 * n.distance:
            matchingPts1.append((pts1[m.queryIdx].pt[0],pts1[m.queryIdx].pt[1]))
            matchingPts2.append((pts2[n.trainIdx].pt[0],pts2[n.trainIdx].pt[1]))
    matchingPts1 = [[int(val[0]), int(val[1])] for val in matchingPts1]
    matchingPts2 = [[int(val[0]), int(val[1])] for val in matchingPts2]
    return matchingPts1, matchingPts2

def hmat_pinv(x, x_prime):
    A = np.ones((len(x) * 2, 8))
    C = np.ones(len(x) * 2)
    # print(A.shape, C.shape)
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
            in_indices = np.where(error <= delta)[0]
            if len(in_indices) > prev_size:
                print("reached", len(in_indices), (1-e)*n_total)
                idx_set = in_indices
                if len(idx_set) > (1 - e) * n_total:
                    outliers = False
                    break
                prev_size=len(in_indices)
            count+=1
        if len(idx_set) > (1-e)*n_total:
            outliers=False
            print("Here")
        else:
            e*=2
            N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
            print("new N", N)
    print("num inliers found:", len(idx_set))
    return idx_set


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
    # error = error @ np.ones((2,1))
    # error = rearrange(error, 'c h -> (c h)')
    return error


def pan(comb, img, H):
    h=img.shape[0]
    w=img.shape[1]
    h_tot=comb.shape[0]
    w_tot=comb.shape[1]
    H = np.linalg.inv(H)
    X,Y=np.meshgrid(np.arange(0,w_tot,1),np.arange(0,h_tot,1))
    pts=np.vstack((X.ravel(), Y.ravel())).T
    pts=np.hstack((pts[:,0:2],pts[:,0:1]*0+1))
    Hpts=H @ pts.T
    Hpts=Hpts/Hpts[2,:]
    Hpts=Hpts.T[:, 0:2].astype('int')
    valid_pts, valid_Hpts = find_valid_pts(pts, Hpts, w-1, h-1)
    for i in range(valid_pts.shape[0]):
        if (comb[valid_pts[i,1], valid_pts[i,0]] != 0).all() == False:
            comb[valid_pts[i, 1], valid_pts[i, 0]]=img[valid_Hpts[i,1], valid_Hpts[i,0]]
    return comb
def find_valid_pts(pts, Hpts, w,h):
    xmin=Hpts[:,0] >=0
    Hpts=Hpts[xmin,:]
    pts=pts[xmin,:]
    xmax=Hpts[:,0]<=w
    Hpts=Hpts[xmax,:]
    pts=pts[xmax,:]
    ymin=Hpts[:,1]>=0
    Hpts=Hpts[ymin,:]
    pts=pts[ymin,:]
    ymax=Hpts[:,1]<=h
    Hpts=Hpts[ymax,:]
    pts=pts[ymax,:]
    return pts, Hpts

