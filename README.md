# Live Emotion Detector 
## By Aya Assaf and Maya Beydoun for EECS 442 Winter 2024

## Project Overview
This project develops a convolutional neural network (CNN) to recognize and classify emotions from facial images using the FER2013 dataset. The project includes a real-time application that uses a webcam to capture facial expressions and overlays corresponding emojis and scores based on the detected emotions.

## Features
- Emotion recognition from images and videos.
- Real-time emotion detection using webcam input.
- Overlays emojis, emotions, and scores in real-time to reflect detected emotions.

## Instructions 
- Download the pretrained model [here](youtube.com) or train and save the model with trainAndEval.ipynb
- Place the model in the same directory as emotion_detector.py
- Run emotion_detector.py 

### To start the real-time emotion detection webcam application
python real_time_detection.py

### To run the emotion detection model on images or videos
python detect_emotions.py --source path/to/image/or/video


### To exit the application use the 'esc' key
