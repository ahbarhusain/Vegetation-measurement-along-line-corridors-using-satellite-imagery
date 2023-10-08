import requests
import json
import pandas as pd
import numpy as np
from requests_toolbelt.adapters.socket_options import TCPKeepAliveAdapter
from threading import Thread
from image_processing import save_image

ALLOWED_EXTENSIONS = set(['tiff', 'tif', 'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG', 'TIF', 'TIFF'])
#URL = [
#        'http://deepforest.druma.com:5555/api/predict-deepforest',
#        'http://hough.druma.com:4444/api/hough-transform'

#        ]
URL=[
    'http://192.168.29.237:5555/api/predict-deepforest',
    'http://192.168.29.237:4444/api/hough-transform'
]




class api_request(Thread):
    
    def __init__ (self, file, url, patch):
        self.result = None
        self.input_image = file
        self.url = url
        self.patch = patch
        super(api_request, self).__init__()

    def run(self):
        keep_alive_conf = TCPKeepAliveAdapter(idle=120, count=20, interval=30)
        sess = requests.session()
        sess.mount('http://', keep_alive_conf)

        if self.url == URL[0]:
            r = sess.post(
                self.url,
                files={'image': open(self.input_image, 'rb')},
                data={'patch_size': self.patch}
                )
            data = json.dumps(r.json())
            self.result = pd.read_json(data, orient='index')
        elif self.url == URL[1]:
            r = sess.post(
                self.url,
                files={'image': open(self.input_image, 'rb')},
                data={'patch_size': self.patch}
                )
            data = json.dumps(r.json())
            self.result = pd.read_json(data, orient='index')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

