import sys
import numpy as np
import cv2 as cv
from scipy import signal
import math
import os
from werkzeug.utils import secure_filename
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

    F = np.zeros( dim, float
)
    L = np.zeros( dim, float
)
    Y = np.zeros( dim, float
)
    T = np.ones( dim, float
)
    Y_AC = np.zeros( dim, float
)
    
    #normalize image
    S = cv.normalize(src.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        F = np.exp(-Alpha_F) * F + V_F * signal.convolve2d(Y, W, mode='same') + S
        L = np.exp(-Alpha_L) * L + V_L * signal.convolve2d(Y, M, mode='same')
        U = F * (1 + Beta * L)
        T = np.exp(-Alpha_T) * T + V_T * Y
        Y = (U>T).astype(float
)
        Y_AC = Y_AC + Y
    
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    return Y_AC

def hough_transform(input_image, app):
    #img = cv.imdecode(np.fromstring(input_image, np.uint8), cv.IMREAD_UNCHANGED)
    img = cv.imread(save_image(input_image))

    Y_AC = pcnn(img)
    edges = cv.Canny((Y_AC*255).astype(np.uint8),100,100,apertureSize = 3)
    lines_pcnn = cv.HoughLines(edges,1,np.pi/180,350)

    if lines_pcnn is not None:
        print(type(lines_pcnn))
        return lines_pcnn
    else:
        return "No Line Detected"

def save_image(img_file):
    filename = secure_filename(img_file.filename)
    img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)