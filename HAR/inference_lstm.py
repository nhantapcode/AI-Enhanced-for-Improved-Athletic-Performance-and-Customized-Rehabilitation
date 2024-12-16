import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import logging

logging.basicConfig(level=logging.DEBUG)

# Label for classification
label = "Warmup...."
n_time_steps = 10
lm_list = deque(maxlen=n_time_steps)

# Mediapipe pose setup
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)
mpDraw = mp.solutions.drawing_utils


# Define the LSTM model to match training
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x, _ = self.lstm4(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # Use the output of the last timestep
        return self.sigmoid(x)


# Load PyTorch model
input_dim = 132  # Number of pose landmarks (33) x 4 (x, y, z, visibility)
hidden_dim = 50
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, output_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


# Helper functions
def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(
        img,
        label,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list, dtype=np.float32)
    lm_list = (
        torch.tensor(lm_list, dtype=torch.float32).unsqueeze(0).to(device)
    )  # Add batch dimension
    with torch.no_grad():
        results = model(lm_list).cpu().numpy()
    if results[0][0] > 0.5:
        label = "HAND SWING"
    elif results[0][0] <= 0.5:
        label = "BODY SWING"
    else:
        label = "UNKNOWN"
    return label


# Main loop
cap = cv2.VideoCapture(0)

warmup_frames = 60
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    frame_count += 1
    if frame_count > warmup_frames:
        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)

            if len(lm_list) == n_time_steps:
                label = detect(model, list(lm_list))

            img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
