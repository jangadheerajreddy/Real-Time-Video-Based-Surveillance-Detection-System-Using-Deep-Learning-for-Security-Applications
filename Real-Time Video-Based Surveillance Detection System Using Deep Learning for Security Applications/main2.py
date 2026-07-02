from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import time
import os
import winsound

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

duration = 700
freq = 900


labelsPath = "obj.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25)


MIN_AREA = 1200



def generate_frames(video_source):

    vs = cv2.VideoCapture(video_source)
    fall_counter = 0          
    abnormal_alert_played = False  

    while True:
        success, frame = vs.read()
        if not success:
            break

        frame = cv2.resize(frame, (900, 650))
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fgmask = fgbg.apply(gray)

       
        _, fgmask_clean = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fall_detected = False

        if contours:
           
            contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)

                if h < w:   # Falls happen when width > height (horizontal body)
                    fall_counter += 1
                else:
                    fall_counter = 0
                    abnormal_alert_played = False

                
                if fall_counter > 6:
                    fall_detected = True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(frame, "ABNORMAL: ABNORMAL DETECTED!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                    if not abnormal_alert_played:
                        winsound.Beep(freq, duration)
                        abnormal_alert_played = True

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Normal Activity", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

      
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:

                    (H, W) = frame.shape[:2]
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

       
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, w, h) = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 255), 2)
                text = "{} {:.2f}".format(LABELS[classIDs[i]],
                                          confidences[i])
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

        
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )




@app.route("/")
def home():
    return render_template("home.html")

@app.route("/concept")
def concept():
    return render_template("concept.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/video_feed_live")
def video_feed_live():
    return Response(generate_frames(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload_video", methods=['POST'])
def upload_video():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return render_template("video_result.html", video_file=file.filename)

@app.route("/video_feed_uploaded/<filename>")
def video_feed_uploaded(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(generate_frames(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
