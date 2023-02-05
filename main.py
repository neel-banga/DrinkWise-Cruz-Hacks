from flask import Flask, render_template, request
import os
import face_model
import cv2
import voice_model

val = None
val2 = None
app = Flask(__name__)


@app.route('/')
def camera():
  return render_template('camera.html')


@app.route('/upload_photo', methods=['POST'])
def upload_photo():
  photo = request.files['photo']
  photo.save(os.path.join(os.getcwd(), photo.filename))

  val = face_model.check_intoxicated(photo.filename)
  print(val)

  return render_template('voice_model.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
  video = request.files['video']
  video_filename = os.path.join(os.getcwd(), video.filename)
  video.save(video_filename)

  phrase = voice_model.get_phrase(2)
  print(phrase)
  with open('phrase.txt', 'w') as f:
    f.write(phrase)

  voice_model.get_wav_file(video.filename)

  val2 = voice_model.check_slurring('test.wav', phrase)

  if ((val + val2)/2) >= 0.6:
    return render_template('not_intoxicated.html')
  else:
    return render_template('intoxicated.html')

@app.route('/ph')
def ph():
  print('hi')
  return render_template('camera.html')

app.run(host='0.0.0.0', port=18)
