import cv2
import os
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np

def extract_frames(video_path, output_folder, frame_interval=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
            
        frame_count += 1
    
    cap.release()
    return extracted_count

def detect_faces(frame_folder, num_frames):
    all_face_locations = []
    
    for i in range(num_frames):
        frame_path = os.path.join(frame_folder, f"frame_{i:04d}.jpg")
        img = cv2.imread(frame_path)
        result = DeepFace.extract_faces(img_path=img, detector_backend='opencv', enforce_detection=False)
        # extract and paint the face locations
        print(result)
        return all_face_locations
    
        
        
def track_faces(all_face_locations):
    tracked_faces = []

    for frame_faces in all_face_locations:
        frame_tracked = []
        for face in frame_faces:
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            center_x = x + w / 2
            center_y = y + h / 2
            frame_tracked.append((center_x, center_y))
        tracked_faces.append(frame_tracked)
    
    return tracked_faces

def plot_dispersion(tracked_faces):
    plt.figure(figsize=(10, 6))
    
    for frame_index, frame_faces in enumerate(tracked_faces):
        for face_index, (x, y) in enumerate(frame_faces):
            plt.scatter(x, y, c='blue', alpha=0.5, label=f'Face {face_index}' if frame_index == 0 else "")
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Dispersion of Face Movement')
    plt.legend()
    plt.grid(True)
    plt.show()

# Paths
video_path = '../videos/sample2.mp4'
output_folder = 'extracted_frames'
frame_interval = 10

# Process video
num_frames = 79
#num_frames = extract_frames(video_path, output_folder, frame_interval)
all_face_locations = detect_faces(output_folder, num_frames)
# tracked_faces = track_faces(all_face_locations)
# plot_dispersion(tracked_faces)
