import os
from flask import request, jsonify, make_response
from werkzeug.utils import secure_filename
from app import app
from hough import hough_transform
#from hough_modified import get_lines

@app.route("/api/hough-transform", methods=["POST"])
def POST_handler():
    if request.method == "POST" :
        input_image = request.files['image']
        results = hough_transform(input_image, app)
        #results = get_lines(input_image, app)
        if isinstance(results, str):
            return jsonify(result=results)
        else:
            return jsonify(lines=results.tolist())

if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port='4444')