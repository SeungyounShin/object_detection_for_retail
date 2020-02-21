from flask import Flask,render_template,request
import os,random,sys
from detect0 import detect
import json

app = Flask(__name__)


@app.route('/')
def mainUI():
  test_images_path = "./static/servertest"
  test_images_list = os.listdir(test_images_path)
  choice_path = test_images_path+"/"+random.choice(test_images_list)
  return render_template("main.html", img_path = choice_path)

@app.route('/calculate', methods = ['POST'])
def calculate():
    path = request.form['img_path']
    detector = detect()
    items = detector.getItems(path)
    return json.dumps(items)

if __name__ == '__main__':
    app.run()
