import cv2
import numpy as np
from tensorflow.keras.models import load_model
from win10toast import ToastNotifier

# Load face detection model and emotion classification model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model("model.h5")

emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

messages = {
    "Happy": "You're smiling! Keep shining ✨",
    "Sad": "You look down. Want to take a break? 💙",
    "Angry": "Let's breathe in... and out 😤",
    "Fear": "Try to relax. You're safe now. 🤍",
    "Neutral": "Keep going. Stay hydrated 💧",
    "Surprise": "Something unexpected? You're doing great 💡",
    "Disgust": "Eww? Close the tab and restart 😅"
}

toaster = ToastNotifier()
cap = cv2.VideoCapture(0)
last_emotion = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        prediction = model.predict(roi)
        emotion_idx = int(np.argmax(prediction))
        emotion = emotion_dict[emotion_idx]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if emotion != last_emotion:
            toaster.show_toast(
                "Mood Check",
                messages.get(emotion, "Everything okay?"),
                duration=5,
                threaded=True
            )
            last_emotion = emotion

    cv2.imshow("Mood Detector", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 