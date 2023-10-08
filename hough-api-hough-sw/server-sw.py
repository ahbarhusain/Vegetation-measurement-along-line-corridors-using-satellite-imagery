import os
from flask import request, jsonify, Response
from werkzeug.utils import secure_filename
from app import app
from hough_sw import get_powerline

@app.route("/api/hough-transform", methods=["POST"])
def POST_handler():
    if request.method == "POST" :
        input_image = request.files['image']
        patch_size = int(request.form['patch_size'])
        results = get_powerline(input_image, app, patch_size)
        return Response(results, mimetype='application/json')

if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port='4444')