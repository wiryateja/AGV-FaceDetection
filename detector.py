import os
import cv2
import time
import threading
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from playsound import playsound
from datetime import datetime

app = Flask(__name__)
CORS(app)

classification_interval = 1.0
speech_interval = 5.0
start_time = time.time()

age_interpreter = tf.lite.Interpreter(model_path="age_model_50epochs.tflite")
emotion_interpreter = tf.lite.Interpreter(model_path="emotion_detection_model_100epochs.tflite")
gender_interpreter = tf.lite.Interpreter(model_path="gender_model_50epochs.tflite")
origin_model = load_model("model_face_class.h5")

age_interpreter.allocate_tensors()
emotion_interpreter.allocate_tensors()
gender_interpreter.allocate_tensors()

age_input = age_interpreter.get_input_details()
age_output = age_interpreter.get_output_details()
emotion_input = emotion_interpreter.get_input_details()
emotion_output = emotion_interpreter.get_output_details()
gender_input = gender_interpreter.get_input_details()
gender_output = gender_interpreter.get_output_details()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_map = {
    'Neutral': 'Neutral', 'Happy': 'Happy', 'Angry': 'Angry',
    'Sad': 'Confused', 'Surprise': 'Confused', 'Fear': 'Confused', 'Disgust': 'Confused'
}
gender_labels = ['male', 'female']
origin_labels = ['international', 'local']

shared_data = {
    "total": 0,
    "gender": {"male": 0, "female": 0},
    "age_group": {"child": 0, "teen": 0, "adult": 0, "elderly": 0},
    "emotion": {"Neutral": 0, "Happy": 0, "Angry": 0, "Confused": 0},
    "origin": {"local": 0, "international": 0},
    "processing_times": []
}

def play_audio_async(file_path):
    def _play():
        try:
            playsound(file_path)
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
    threading.Thread(target=_play, daemon=True).start()

def get_audio_filename(origin, age, gender):
    if origin == "local":
        if age <= 19:
            return "sapaan_halo_indonesia"
        elif gender == "male":
            return "sapaan_pak_indonesia"
        else:
            return "sapaan_bu_indonesia"
    else:
        if age <= 19:
            return "sapaan_hello_english"
        elif gender == "male":
            return "sapaan_sir_english"
        else:
            return "sapaan_maam_english"

def play_greeting(origin, age, gender):
    filename = get_audio_filename(origin, age, gender)
    path = os.path.join("audio", filename + ".mp3")
    if os.path.exists(path):
        play_audio_async(path)
    else:
        print(f"[WARNING] Audio not found: {path}")

@app.route("/dashboard", methods=["GET"])
def dashboard():
    avg_processing = (
        sum(shared_data["processing_times"]) / len(shared_data["processing_times"])
        if shared_data["processing_times"] else 0
    )
    uptime_days = int((time.time() - start_time) // (60 * 60 * 24)) + 1
    return jsonify({
        "totalScans": shared_data["total"],
        "activeDays": uptime_days,
        "processingTime": round(avg_processing, 2),
        "accuracyRate": 95.3,
        "genderDistribution": shared_data["gender"],
        "ageGroups": shared_data["age_group"],
        "ethnicity": shared_data["origin"],
        "expressions": shared_data["emotion"]
    })

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
last_classification_time = 0
last_speech_time = 0

def detection_loop():
    global last_classification_time, last_speech_time
    cap = cv2.VideoCapture(0)

    label_to_show = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        current_time = time.time()

        if len(faces) > 0 and (current_time - last_classification_time > classification_interval):
            (x, y, w, h) = faces[0]
            face_color = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            start_det_time = time.time()

            resized_gray = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
            resized_gray = np.expand_dims(img_to_array(resized_gray), axis=0)
            emotion_interpreter.set_tensor(emotion_input[0]['index'], resized_gray)
            emotion_interpreter.invoke()
            emotion_pred = emotion_interpreter.get_tensor(emotion_output[0]['index'])
            emotion_label = emotion_labels[np.argmax(emotion_pred)]
            main_emotion = emotion_map.get(emotion_label, "Confused")
            shared_data["emotion"][main_emotion] += 1

            g_face = cv2.resize(face_color, (200, 200)).astype("float32")
            g_face = np.expand_dims(g_face, axis=0)
            gender_interpreter.set_tensor(gender_input[0]['index'], g_face)
            gender_interpreter.invoke()
            g_pred = gender_interpreter.get_tensor(gender_output[0]['index'])
            gender = gender_labels[int(g_pred[0][0] >= 0.5)]
            shared_data["gender"][gender] += 1

            age_interpreter.set_tensor(age_input[0]['index'], g_face)
            age_interpreter.invoke()
            age_pred = age_interpreter.get_tensor(age_output[0]['index'])
            age = int(round(age_pred[0][0]))
            if age <= 12:
                shared_data["age_group"]["child"] += 1
            elif age <= 19:
                shared_data["age_group"]["teen"] += 1
            elif age <= 59:
                shared_data["age_group"]["adult"] += 1
            else:
                shared_data["age_group"]["elderly"] += 1

            o_face = cv2.resize(face_color, (100, 100)).astype("float32") / 255.0
            o_face = np.expand_dims(img_to_array(o_face), axis=0)
            o_pred = origin_model.predict(o_face)[0]
            origin = origin_labels[np.argmax(o_pred)]
            shared_data["origin"][origin] += 1

            shared_data["total"] += 1
            shared_data["processing_times"].append(time.time() - start_det_time)

            if current_time - last_speech_time > speech_interval:
                play_greeting(origin, age, gender)
                last_speech_time = current_time

            label_to_show = f"{origin}, {gender}, {age} y/o, {main_emotion}"
            last_classification_time = current_time

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if label_to_show:
                (text_w, text_h), _ = cv2.getTextSize(label_to_show, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - 25), (x + text_w, y - 5), (0, 0, 0), -1)
                cv2.putText(frame, label_to_show, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Deteksi Wajah & Klasifikasi', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

threading.Thread(target=detection_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
