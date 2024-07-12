import cv2
import torch
import datetime
from fer_pytorch.fer import FER

fer = FER()
print(fer.device)

fer.get_pretrained_model(model_name="resnet34")

# Load your video
video_path = 'videos/test_2.mp4'
cap = cv2.VideoCapture(video_path)

path_to_output = f'tests/results{datetime.datetime.now().timestamp()}'
fer.analyze_video(path_to_video=video_path, path_to_output=path_to_output)
