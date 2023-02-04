from flask import Flask, render_template, request
import os

app = Flask(__name__)


@app.route('/')
def camera():
  return render_template("camera.html")


@app.route('/upload_photo', methods=['POST'])
def upload_photo():
  photo = request.files['photo']
  photo.save(os.path.join(os.getcwd(), photo.filename))
  return render_template("upload.html")
  return "Photo uploaded!"


app.run(host='0.0.0.0', port=81)
