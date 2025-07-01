Real-Time Emotion-Based Pop-Up Assistant

This is a real-time computer vision application that uses your webcam to detect facial emotions and displays a system notification based on the detected mood. It is useful for building personal awareness, mental wellness tools, or mood-based assistant systems. This project runs in VS Code and uses a Convolutional Neural Network (CNN) trained on the FER2013 dataset.


FEATURES
- Detects 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- Uses a CNN model trained on the FER2013 dataset.
- Real-time face and emotion detection using OpenCV.
- System notifications appear when the detected emotion changes.
- Two notification options:
  - `plyer`: Shows notifications in the Windows Action Center
  - `win10toast`: Shows toast pop-up on-screen (Windows)

FOLDER STRUCTURE
emotion-popup/

├── fer2013.csv # Emotion dataset (Kaggle FER2013)

├── train_model.py # Script to train the emotion detection model

├── model.h5 # Saved trained model

├── main.py # Real-time detection using plyer (Action Center)

├── main1.py # Real-time detection using win10toast (toast popup)

├── README.md # Documentation


VIRTUAL ENVIRONMENT (`.venv`)

This project uses a Python virtual environment stored in the `.venv` folder. It helps manage dependencies locally without affecting your global Python installation.

To create a virtual environment (if not already created):

```bash
python -m venv .venv
```

To activate the environment:

On Windows:
```bash
.venv\Scripts\activate
```
On macOS/Linux:
```bash
source .venv/bin/activate
```
After activation, install all required packages:

```bash
pip install -r requirements.txt
```


DATASET
Download the FER2013 Dataset from Kaggle and place fer2013.csv inside the project folder. 

The `csv file` can be downloaded from here: https://www.kaggle.com/datasets/deadskull7/fer2013

The `dataset` can be downloaded from here: https://www.kaggle.com/datasets/msambare/fer2013/data

Training the Model
Run the following script to train a CNN on the FER2013 dataset: python train_model.py
This will generate model.h5, a trained model file used for real-time emotion prediction.


RUNNING THE APPLICATION
- Choose one of the following based on your operating system and preferences:

Option 1: main.py (plyer - Action Center Notifications)
This version uses the plyer library. Notifications will appear in the Action Center on Windows.
python main.py

Option 2: main1.py (win10toast - Visible Toast Popups)
This version uses the win10toast library to show toast notifications (visible pop-up) on Windows.
python main1.py


HOW IT WORKS
1. Opens your system webcam.
2. Detects faces using Haar cascades.
3. Extracts the face region, resizes and normalizes it.
4. Predicts the emotion using the trained CNN model.
5. Displays a notification message when a new emotion is detected.

NOTES

No GPU is required — runs on most laptops.

Ensure your webcam is connected and working.

Notifications may not appear if you're in full-screen mode (e.g., games or video players).
