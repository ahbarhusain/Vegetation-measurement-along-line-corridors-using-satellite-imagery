from flask import Flask

UPLOAD_FOLDER = './static/uploads/'

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "this_is_your_key"
