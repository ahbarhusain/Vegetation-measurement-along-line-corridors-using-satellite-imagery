import sys
import numpy as np
import cv2 as cv
from scipy import signal
import math
import os
import pandas as pd
import slidingwindow as sw
from werkzeug.utils import secure_filename
from tqdm import tqdm
from app import app

Alpha_F = 0.1
Alpha_L = 1.0
Alpha_T = 0.3

V_F = 0.5
V_L = 0.2
V_T = 20.0

Num = 10
Beta = 0.1

W = [0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,]
W = np.array(W, float).reshape((3, 3))
M = W

global F
global L
global Y
global T
global Y_AC

def pcnn(input_image):
    src = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    
    dim = src.shape

    F = np.zeros( dim, float)
    L = np.zeros( dim, float)
    Y = np.zeros( dim, float)
    T = np.ones( dim, float)
    Y_AC = np.zeros( dim, float)
    
    S = cv.normalize(src.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    
    for cont in range(Num):
        F = np.exp(-Alpha_F) * F + V_F * signal.convolve2d(Y, W, mode='same') + S
        L = np.exp(-Alpha_L) * L + V_L * signal.convolve2d(Y, M, mode='same')
        U = F * (1 + Beta * L)
        T = np.exp(-Alpha_T) * T + V_T * Y
        Y = (U>T).astype(float
)
        Y_AC = Y_AC + Y
    
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    return Y_AC


def hough_transform(input_image, hough_thres):
    Y_AC = pcnn(input_image)
    edges = cv.Canny((Y_AC*255).astype(np.uint8),100,100,apertureSize = 3)
    lines_pcnn = cv.HoughLines(edges,1,np.pi/180,hough_thres)
    return lines_pcnn


def get_powerline(input_image, app, patch=500):
    img = cv.imread(save_image(input_image))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    if (img.shape[0] >= 3000) and (img.shape[0] >= 3000):
        windows = generate_sw(img,patch,0)
        line_df = pd.DataFrame(columns=["window", "pt1", "pt2"])

        for index, window in enumerate(tqdm(windows)):
            crop = img[windows[index].indices()]
            result = detect_powerline(crop)
            
            if result is None:
                continue
            else:
                xmin, ymin, xmax, ymax = windows[index].getRect()
                pt1,pt2 = convert_to_cartesian(result,
                                            img_width=patch,
                                            img_height=patch,
                                            xmin=xmin,
                                            ymin=ymin)
                for i in range(len(pt1)):
                    new_row = [index, pt1[i], pt2[i]]
                    line_df = pd.concat([line_df, pd.DataFrame([new_row], columns=line_df.columns)], ignore_index=True)

        
        print("number of lines detected",line_df.shape[0])

    else:
        num_line = 10000
        line_df = pd.DataFrame(columns=["pt1", "pt2"])

        for kernel_size in tqdm(range(3,15,2)):
            img = cv.bilateralFilter(img, 3, 250, 250)
            i = 100
            while (i <= 400):    
                lines = hough_transform(img, i)
                if lines is None:
                    break
                if (lines.shape[0] < num_line) and (lines.shape[0] >= 2):
                    num_line = lines.shape[0]
                    result = lines
                i = i + 20
            if (lines is None) and (result.shape[0] <= 3) and (result.shape[0] >= 1):
                break
        
        pt1,pt2 = convert_to_cartesian(result)

        for i in range(len(pt1)):
            new_row = [pt1[i], pt2[i]]
            line_df = pd.concat([line_df, pd.DataFrame([new_row], columns=line_df.columns)], ignore_index=True)


    return line_df.to_json(orient='index')
    

def detect_powerline(img):
    num_line = 10000
    lines = None
    result = None
    kernel_size = 3
    
    while (kernel_size <= 13):
        img = cv.bilateralFilter(img, 3, 250, 250)
        i = 100
        while (i <= 400):    
            lines = hough_transform(img, i)
            if lines is None:
                break
            if (lines.shape[0] < num_line) and (lines.shape[0] >= 2):
                num_line = lines.shape[0]
                result = lines
            i = i + 20
        kernel_size = kernel_size + 2

        if (lines is None) and (result is None):
            break
        elif (lines is None) and (result.shape[0] <= 3) and (result.shape[0] >= 1):
            continue
            
    if (lines is None) and (result is None):
        return None
    elif (lines is None) and (result.shape[0] <= 5) and (result.shape[0] >= 2):
        return result


def convert_to_cartesian(hough_res, img_width=2000, img_height=2000, xmin=0, ymin=0):
    pt1_list = []
    pt2_list = []
    for i in range(0, len(hough_res)):
        rho = hough_res[i][0][0]
        theta = hough_res[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = [int(x0 + img_width*(-b) + xmin), int(y0 + img_height*(a)) + ymin]
        pt2 = [int(x0 - img_width*(-b) + xmin), int(y0 - img_height*(a)) + ymin]
        pt1_list.append(pt1)
        pt2_list.append(pt2)
    
    return pt1_list, pt2_list


def generate_sw(np_image, patch_size, patch_overlap):
    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))
    windows = sw.generate(np_image,
                          sw.DimOrder.HeightWidthChannel,
                          patch_size,
                          patch_overlap)
    return windows  


def save_image(img_file):
    filename = secure_filename(img_file.filename)
    img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)