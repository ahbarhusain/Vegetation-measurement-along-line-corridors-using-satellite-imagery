import numpy as np
import pandas as pd
import cv2 as cv
import os
from werkzeug.utils import secure_filename
#from deepforest import deepforest
from deepforest import get_data
from deepforest import main 


from app import app

def load_model():
    try:
        #model = deepforest.deepforest(saved_model="deepforest-model.h5")
        model = main.deepforest(saved_model="deepforest-model.h5")
        print("Trained Model loaded")
    except:
        model = main.deepforest()
        model.use_release()
        print("Prebuilt Model loaded")
    return model

def predict(input_image, model, patch=700):
    img_path = save_image(input_image)
    bounding_boxes = model.predict_tile(    
                        raster_path=img_path, 
                        return_plot=False,
                        patch_overlap=0.3, 
                        iou_threshold=0.2, 
                        patch_size=patch
                    )

    bounding_boxes["xcenter"] = get_x_center(bounding_boxes["xmin"], bounding_boxes["xmax"])
    bounding_boxes["ycenter"] = get_y_center(bounding_boxes["ymin"], bounding_boxes["ymax"])

    #os.remove(input_image)

    return bounding_boxes.to_json(orient='index')

def get_x_center(xmin, xmax):
    return (xmin + xmax) * 0.5

def get_y_center(ymin, ymax):
    return (ymin + ymax) * 0.5

def save_image(img_file):
    filename = secure_filename(img_file.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(img_path)

    img = cv.imread(img_path)
    img = img[:,:,:3]
    cv.imwrite(img_path, img)

    return img_path
 