import numpy as np
import pandas as pd
import cv2 as cv
import os
import math
from PIL import Image
from werkzeug.utils import secure_filename
from app import app

DANGER_DIST = 15
CAUTION_DIST = 25

def load_image_as_np(img_file):
    img = cv.imread(img_file)
    return img


def save_image(img_file):
    filename = secure_filename(img_file.filename)
    img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)


def save_res_image(np_image, filename):
    cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'result-'+filename), np_image)
    return 'result-' + filename 


def draw_line(image, rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = [int(x0 + 2000*(-b)), int(y0 + 2000*(a))]
    pt2 = [int(x0 - 2000*(-b)), int(y0 - 2000*(a))]
    cv.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]), (255,0,0), 2, cv.LINE_AA)
    return pt1,pt2


def draw_box(image, box, color=[0,255,0], thickness=1):
    b = np.array(box).astype(int)
    cv.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv.LINE_AA)


def get_shortest_distance(p, a, b, resolution=0.5):
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:,0], d_ba[:,1]).reshape(-1,1)))
    s = np.multiply(a-p, d).sum(axis=1)
    t = np.multiply(p-b, d).sum(axis=1)
    h = np.maximum.reduce([s, t, np.zeros(len(s))])
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    distance = np.hypot(h,c).min()
    return distance * resolution


def get_box_color(center_point, line_start, line_end, spat_res):
    color = []
    for i in range(0, center_point.shape[0]):
        distance = get_shortest_distance(center_point[i], line_start, line_end, spat_res)
        if ( distance >= 0) and (distance <= DANGER_DIST):
            color.append('red')
        elif (distance > DANGER_DIST) and (distance <= CAUTION_DIST):
            color.append('yellow')
        else:
            color.append('green')
    return color



