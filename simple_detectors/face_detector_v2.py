from deepface import DeepFace
import pandas as pd
from tqdm import tqdm
import cv2
import os
import matplotlib.pyplot as plt
import ast
from sixdrepnet import SixDRepNet
from google.colab.patches import cv2_imshow

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

    def __init__(self, distance_metric, frame_window ,facial_tracker_window, backend_analyze, backend_find, model_find, db_path):
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
        self.results = pd.DataFrame(columns=['student_id', 'Frame', 'Emotions', 'Main Emotion', 'Face Position'])

        self.head_pose_model = SixDRepNet()

    def detect_faces(self, frame):

        analysis_results = DeepFace.analyze(frame, detector_backend=self.backend_analyze, enforce_detection=True, actions=['emotion'], silent=True)
        # store the detected faces in a directory for further processing
        for res in analysis_results:

            if res['face_confidence'] != 1.0:
                # print("Face not detected")
                continue

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
                if(dfs[0].shape[0] == 0):
                    cv2.imwrite(f"{self.db_path}/{self.face_counter}.jpg", face_img)
                    person = f"student_{self.face_counter}"
                    self.faces_iter[person] = {'iter': 1}
                    self.face_counter+=1
                else:
                    student_number = dfs[0].iloc[0].identity.split('/')[-1].replace('.jpg', '').split('_')[0]
                    person =f"student_{student_number}"
                    if (self.frame_count % self.facial_tracker_window == 0):
                        cv2.imwrite(f"{self.db_path}/{student_number}_iter{self.faces_iter[person]['iter']}.jpg", face_img)
                        self.faces_iter[person]['iter']+=1

            else:
                cv2.imwrite(f"{self.db_path}/{self.face_counter}.jpg", face_img)
                person = f"student_{self.face_counter}"
                self.faces_iter[person] = {'iter': 1}
                self.face_counter+=1

            
            head_pose_result = self.extract_head_pose(face_img, verbose=True)

            # Analyzing the emotion of the cropped face image
            cv2.rectangle(frame, (x, y), (x+w, y+h), (240, 0, 0), 2)
            cv2.putText(frame, res['dominant_emotion'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
            cv2.putText(frame, person, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
            cv2.putText(frame, head_pose_result[0], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)

            self.results.loc[len(self.results)] = [person, self.frame_count, res['emotion'], res['dominant_emotion'], res['region'], head_pose_result]

    def extract_head_pose(self, face_img, verbose=False):
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
          self.head_pose_model.draw_axis(face_img, yaw, pitch, roll)
          print(f"Pitch: {pitch}, Yaw: {yaw}, Roll: {roll}")
        
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
                    cv2_imshow(frame)
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


if __name__=="__main__":

    db_path = 'faces_db'

    if not os.path.exists(db_path):
        os.makedirs(db_path)
    else:
        for file in os.listdir(db_path):
            os.remove(f"{db_path}/{file}")

    new_analysis = run_analysis(
        distance_metric = METRICS_LIST[1],
        frame_window=10,
        facial_tracker_window = 20,
        backend_analyze = BACKENDS[4],
        backend_find = BACKENDS[0],
        model_find = MODELS[2],
        db_path=db_path)
    new_analysis.process_video('sample2.mp4')
    new_analysis.export_results()
    plot_emotion_evolution('results.csv')


