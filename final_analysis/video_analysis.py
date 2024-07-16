from deepface import DeepFace
import pandas as pd
from tqdm import tqdm
import cv2
import os
import matplotlib.pyplot as plt
import ast
from sixdrepnet import SixDRepNet
import argparse
import time

BACKENDS = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'fastmtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]


MODELS = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]

METRICS_LIST = ["cosine", "euclidean", "euclidean_l2"]

class run_analysis:

    def __init__(self, distance_metric, frame_window ,facial_tracker_window, backend_analyze, backend_find, model_find, db_path, verbose):
        self.distance_metric = distance_metric
        self.frame_window = frame_window
        self.facial_tracker_window = facial_tracker_window
        self.backend_analyze = backend_analyze
        self.backend_find = backend_find
        self.model_find = model_find
        self.db_path = db_path
        self.face_counter = 0
        self.frame_count = 0
        self.faces_iter = {}
        self.verbose = verbose
        self.results = pd.DataFrame(columns=['student_id', 'Frame', 'Emotions', 'Main Emotion', 'Region', 'Face Position'])

        self.head_pose_model = SixDRepNet()

    def detect_faces(self, frame):

        analysis_results = DeepFace.analyze(frame, detector_backend=self.backend_analyze, enforce_detection=True, actions=['emotion'], silent=False)
        # store the detected faces in a directory for further processing
        for res in analysis_results:

            if res['face_confidence'] < 0.9:
                print("Face not detected with confidence" + str(res['face_confidence']))
                break

            x, y, w, h = list(res['region'].values())[:4]
            face_img = frame[y:y+h, x:x+w]

            person = 'unknown'

            #face recognition
            if any(file for file in os.listdir(self.db_path) if not file.endswith('.pkl')):
                dfs = DeepFace.find(face_img,
                                    db_path = self.db_path,
                                    distance_metric = self.distance_metric,
                                    model_name=self.model_find,
                                    detector_backend=self.backend_find,
                                    silent=True,
                                    enforce_detection=False)
                
                if len(dfs) > 0:
                    student_number = dfs[0].iloc[0].identity.split('/')[-1].replace('.jpg', '').split('_')[0]
                    person =f"student_{student_number}"

            # Extract head pose
            head_pose_result = []
            head_pose_result = self.extract_head_pose(face_img, paintAxis=True)

            # Painting the rectangle and putting the text
            if(self.verbose):   
                cv2.rectangle(frame, (x, y), (x+w, y+h), (240, 0, 0), 2)
                cv2.putText(frame, res['dominant_emotion'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
                cv2.putText(frame, person, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
                cv2.putText(frame, head_pose_result[0], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)

            self.results.loc[len(self.results)] = [person, self.frame_count, res['emotion'], res['dominant_emotion'], res['region'], {'pitch': float(head_pose_result[1]), 'yaw': float(head_pose_result[2]), 'roll': float(head_pose_result[3])}]

    def extract_head_pose(self, face_img, verbose=False, paintAxis=False):
        pitch, yaw, roll = self.head_pose_model.predict(face_img)

        if pitch > 30:
          text = 'Top'
          if yaw > 30:
            text = 'Top Right'
          elif yaw < -30:
            text = 'Top Left'
        elif pitch < -30:
          text = 'Bottom'
          if yaw > 30:
            text = 'Bottom Right'
          elif yaw < -30:
            text = 'Bottom Left'
        elif yaw > 30:
          text = 'Right'
        elif yaw < -30:
          text = 'Left'
        else:
          text = 'Forward'
        
        if verbose:
          print(f"Text: {text}, Pitch: {pitch}, Yaw: {yaw}, Roll: {roll}")
        
        if paintAxis:
            self.head_pose_model.draw_axis(face_img, yaw, pitch, roll)
        
        return [text, pitch, yaw, roll]
            

    # Detect and track faces
    def process_video(self, video_path):
        total_frame_count = 0;
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            total_frame_count += 1
            ret, frame = cap.read()
            if not ret:
                break

        cap = cv2.VideoCapture(video_path)
        pbar = tqdm(total = total_frame_count, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', ncols=100, desc="Processing video", unit="frames")

        while(cap.isOpened()):
            pbar.update(1)
            ret, frame = cap.read()
            if not ret:
                break
            # Increment frame_count
            self.frame_count += 1
            if(self.frame_count % self.frame_window == 0):
                try:
                    self.detect_faces(frame)
                    if(self.verbose):
                        cv2.imshow("Frame", frame)
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        pbar.close()
        cv2.destroyAllWindows()

    def export_results(self):
        self.results.to_csv('results.csv', index=False)
    # Convert string representation of dictionary to dictionary


def parse_dict(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return {}

def plot_emotion_evolution(data_path):
# Read data into DataFrame
    df = pd.read_csv(data_path)
    # Parse Emotions column to dictionary
    df['Emotions'] = df['Emotions'].apply(parse_dict)
    # Group data by student ID
    grouped = df.groupby('student_id')
    # Calculate number of rows and columns for the grid
    num_students = len(grouped)
    num_cols = 2
    num_rows = -(-num_students // num_cols)  # Ceiling division to ensure enough rows
    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))
    # Plot emotion evolution for each student
    for i, (student, group) in enumerate(grouped):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]
        ax.set_title(f'Emotion Evolution for {student}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Emotion Probability (%)')
        for emotion in group['Emotions'].iloc[0].keys():
            ax.plot(group['Frame'], group['Emotions'].apply(lambda x: x.get(emotion, 0)), label=emotion)
        ax.legend()
    # Remove any empty subplots
    for i in range(num_students, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].remove()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run face analysis on a video.")
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--output_path', type=str, default='results.csv', help='Path to the output file')
    parser.add_argument('--db_path', type=str, default='faces_db', help='Path to the database directory')
    parser.add_argument('--distance_metric', type=str, choices=METRICS_LIST, default='euclidean_l2', help='Distance metric for face recognition')
    parser.add_argument('--frame_window', type=int, default=60, help='Number of frames to skip between analyses')
    parser.add_argument('--facial_tracker_window', type=int, default=3, help='Number of times to skip for facial tracking updates')
    parser.add_argument('--backend_analyze', type=str, choices=BACKENDS, default='retinaface', help='Backend for face analysis')
    parser.add_argument('--backend_find', type=str, choices=BACKENDS, default='opencv', help='Backend for face finding')
    parser.add_argument('--model_find', type=str, choices=MODELS, default='Facenet512', help='Model for face finding')
    parser.add_argument('--verbose', type=bool, default=False, help='Print verbose output and show images')
    parser.add_argument('--clear_db', type=bool, default=False, help='Clear the database directory before running the analysis')

    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        os.makedirs(args.db_path)

    if args.clear_db:
        for file in os.listdir(args.db_path):
            os.remove(f"{args.db_path}/{file}")

    try: 
        with open(args.output_path, 'x') as file: 
            file.write("student_id,Frame,Emotions,Main Emotion,Region,Face Position\n") 
    except FileExistsError: 
        print(f"The file '{args.output_path}' already exists.") 

    t=time.time()

    new_analysis = run_analysis(
        distance_metric=args.distance_metric,
        frame_window=args.frame_window,
        facial_tracker_window=args.frame_window*args.facial_tracker_window,
        backend_analyze=args.backend_analyze,
        backend_find=args.backend_find,
        model_find=args.model_find,
        db_path=args.db_path,
        verbose=args.verbose
    )

    new_analysis.process_video(args.video_path)
    new_analysis.export_results()

    print(f"Time to process: {int(time.time() - t)} seconds")

    if args.verbose:
        plot_emotion_evolution(args.output_path)

