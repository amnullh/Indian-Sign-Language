from flask import Flask, render_template, Response, jsonify, send_file
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from string import ascii_uppercase
import cv2
import time
import os
import math
import numpy as np

app = Flask(__name__)

alpha_dict = {}
j=0
for i in ascii_uppercase:
   alpha_dict[j] = i
   j = j + 1

# model = keras.models.load_model("model-all1-alpha.h5")

alp_predict = " "
word =[]
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

file_path = 'camera_testing_report.txt'

# labels = ["1", "2", "3", "A", "B", "C", "D"]

labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def generate_frames():
    global alp_predict
    cap = cv2.VideoCapture(0)

    # -----------------------------------Module-----------------------------------

    while True:
        print("Check 1")
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if len(hands) == 2:  # Check if two hands are detected
            hand1, hand2 = hands  # Extract information about both hands
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']

            print("hand 2")

            # Create a bounding box that encompasses both hands
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            print("Check 2 for hand 2")

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=True)
            print(prediction, index)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                          (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            alp_predict = labels[index]

        if len(hands) == 1:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=True)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=True)

            print(prediction, index)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                                  (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255),
                                2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255),
                                  4)

            alp_predict = labels[index]

            print("Check 3")

            print(alp_predict)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', imgOutput)
            imgOutput = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + imgOutput + b'\r\n')


    # camera.set(cv2.CAP_PROP_FPS, 30)  # Set the desired frame rate (adjust as needed)
    # camera.set(3, 640)  # Width
    # camera.set(4, 480)  # Height
    #
    # while True:
    #     success, frame = camera.read()
    #     frame = cv2.flip(frame, 1)
    #     cv2.rectangle(frame, (319, 9), (620 + 1, 309), (0, 255, 0), 1)
    #     roi = frame[10:300, 320:620]
    #     # cv2.imshow("Frame", frame)
    #     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #     gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    #     smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9,
    #                                        2.8)
    #     ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #     # cv2.imshow("BW", final_image)    comment
    #     final_image = cv2.resize(final_image, (128, 128))
    #     final_image = np.reshape(final_image, (1, final_image.shape[0], final_image.shape[1], 1))
    #     pred = model.predict(final_image)
    #     alp_predict = alpha_dict[np.argmax(pred)]
    #     print(alp_predict)
    #
    #     if not success:
    #         break
    #     else:
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #     yield(b'--frame\r\n'
    #                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/module')
def module():
    return render_template("module.html")


@app.route('/test')
def test():
    return render_template("test.html")


@app.route('/learn')
def learn():
    return render_template("learn.html")


@app.route('/predict')
def predict():
    global alp_predict
    time.sleep(1)
    word.append(alp_predict)
    print(word)
    word_data = ' '.join(word)
    with open(file_path, 'w') as f:
        f.write(word_data)
    return jsonify({'prediction': word_data})


@app.route('/pred_ans')
def pred_ans():
    global alp_predict
    return jsonify({'prediction': alp_predict})

# @app.route('/create_doc')
# def create_doc():
#     pass


@app.route('/download_report')
def download_report():
    # Check if the file exists
    if os.path.exists(file_path):
        # Send the file for download
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found"


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, port=5001)