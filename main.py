from flask import Flask, render_template, request
import os
import face_model


val = None
app = Flask(__name__)


@app.route('/')
def camera():
  return render_template("camera.html")


@app.route('/upload_photo', methods=['POST'])
def upload_photo():
  photo = request.files['photo']
  photo.save(os.path.join(os.getcwd(), photo.filename))

  val = face_model.check_intoxicated(photo.filename)
  print(val)

  return render_template('voice_model.html')

  '''
  if val == 0:
    return render_template('not_intoxicated.html')
  else:
    return render_template('intoxicated.html')
  '''

app.run(host='0.0.0.0', port=81)
