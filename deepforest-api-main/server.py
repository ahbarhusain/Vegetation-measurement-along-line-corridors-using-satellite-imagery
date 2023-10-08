import tensorflow as tf
from flask import request, jsonify, make_response, Response
from app import app
from predict_deepforest import load_model, predict   

model = load_model()
#graph = tf.get_default_graph()

@app.route("/api/predict-deepforest", methods=["POST"])
def POST_handler():
    if request.method == "POST":
        input_image = request.files['image']
        patch_size = int(request.form['patch_size'])
        results = predict(input_image, model, patch_size)
        print(type(results))
        return Response(results, mimetype='application/json')


if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port='5555')