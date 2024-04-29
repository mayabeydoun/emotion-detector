import cv2
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F
import argparse
import os

class Network(nn.Module):
    def __init__(self, num_classes=7):  # 7 emotion classes for FER2013
        super(Network, self).__init__()
     
        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(0.25)
        )
        
        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(0.25)
        )
        
        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(0.25)
        )
        
        # Fourth convolutional block
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Dropout(0.5)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.fc_layers(x)
        return x


    
# load emoji images
emojis = {
    "Angry": cv2.imread('emojis/angry.png', cv2.IMREAD_UNCHANGED),
    "Disgust": cv2.imread('emojis/disgust.png', cv2.IMREAD_UNCHANGED),
    "Fear": cv2.imread('emojis/fear.png', cv2.IMREAD_UNCHANGED),
    "Happy": cv2.imread('emojis/happy.png', cv2.IMREAD_UNCHANGED),
    "Sad": cv2.imread('emojis/sad.png', cv2.IMREAD_UNCHANGED),
    "Surprise": cv2.imread('emojis/surprise.png', cv2.IMREAD_UNCHANGED),
    "Neutral": cv2.imread('emojis/neutral.png', cv2.IMREAD_UNCHANGED)
}

# overlay emoji on screen
def overlay_emoji(x, y, emoji, background):

    emoji = cv2.resize(emoji, (100, 100))
    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_emoji


    start_y = max(y - 100, 0)
    start_x = max(x - 50, 0)
    end_y = start_y + emoji.shape[0]
    end_x = start_x + emoji.shape[1]

    for c in range(0, 3):
        background[start_y:end_y, start_x:end_x, c] = (alpha_emoji * emoji[:, :, c] +
                                                      alpha_background * background[start_y:end_y, start_x:end_x, c])


# command line arguments
# default is webcam(0) , can also use photo or video with --source
parser = argparse.ArgumentParser(description="Run the emotion detection model on video or image input.")
parser.add_argument('--source', default=0, help="Video source, default is webcam (0). Use file path for video/image file.")
args = parser.parse_args()

# check if the source is a file and if it is an image
is_image = False
if not str(args.source).isdigit():
    file_extension = os.path.splitext(args.source)[1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        is_image = True

# init video capture or video file
if is_image:
    image = cv2.imread(args.source)
    if image is None:
        print("Error loading image")
        exit()
    cap = None
else:
    cap = cv2.VideoCapture(args.source if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

# load the trained model, train and save this model from trainAndEval.ipynb
model = torch.load('complete_model1.pth', map_location=torch.device('cpu'))
model.eval()

# haarcascade for face detection, frontal face 
haarcascade_path = 'haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(haarcascade_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    face_images = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        face_images.append(roi_resized)
    return faces, face_images, img

if is_image:
    faces, face_images, processed_image = face_detector(image)
    for i, face_img in enumerate(face_images):
        roi = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        roi = roi.to('cpu')
        with torch.no_grad():
            prediction = model(roi)
            probabilities = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
            predicted_class = torch.argmax(probabilities)
            label = emotion_labels[predicted_class.item()]
            confidence = probabilities[predicted_class.item()].item()
        (x, y, w, h) = faces[i]
        label_position = (x + int(w/2), y + h + 25)
        cv2.putText(processed_image, f"{label}: {confidence:.2f}%", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        label = emotion_labels[predicted_class.item()]
        emoji_img = emojis[label]  
        overlay_emoji(x, y, emoji_img, image)  
    cv2.imshow('Emotion Detector', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # video will loop intil 'esc' key clicked
            continue
        frame = cv2.flip(frame, 1)
        faces, face_images, image = face_detector(frame)
        for i, face_img in enumerate(face_images):
            roi = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            roi = roi.to('cpu')
            with torch.no_grad():
                prediction = model(roi)
                probabilities = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
                predicted_class = torch.argmax(probabilities)
                label = emotion_labels[predicted_class.item()]
                confidence = probabilities[predicted_class.item()].item()
            (x, y, w, h) = faces[i]
            label_position = (x + int(w/2), y + h + 25)
            cv2.putText(image, f"{label}: {confidence:.2f}%", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            label = emotion_labels[predicted_class.item()]
            emoji_img = emojis[label]  
            overlay_emoji(x, y, emoji_img, image)  
        # exit at any time using 'esc' key 
        cv2.imshow('Emotion Detector', image)
        if cv2.waitKey(1) & 0xFF == 27: 
            break
    cap.release()
    cv2.destroyAllWindows()