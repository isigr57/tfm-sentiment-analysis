import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import json
from datetime import datetime
import sys
import matplotlib.pyplot as plt

def initialize_firebase():
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate('./serviceAccount.json')
    firebase_admin.initialize_app(cred)

def create_session_document(file_path, teacher_id):
    # Load the CSV file
    data = pd.read_csv(file_path)
    # Extract unique student_ids
    student_ids = data['student_id'].unique()
    # Create references for students
    student_references = [db.document(f'students/{student_id}') for student_id in student_ids]
    # Create the session document
    session_doc = {
        'name': "Class",
        'createdAt': datetime.now(),
        'students': [],
        'teacherId': teacher_id,
        'sessionData': {
            'mainEmotion': extractMainEmotion(data),
            'overallAttention': float("{:.2f}".format(buildAttentionDataFrame(data)['attention'].mean())),
            'emotionRadar': radarEmotion(data),
            'attentionOverTime': attentionOverTime(data)
        }
    }
    # Add the session document to Firestore
    db.collection('sessions').add(session_doc)
    #print(session_doc)
    print("Session document created successfully.")

def extractMainEmotion(data):
    emotions_count = data['Main Emotion'].value_counts()
    return emotions_count.idxmax()

def buildAttentionDataFrame(data):
    # Function to parse the Face Position column
    def parse_face_position_json(face_position):
        pos_dict = json.loads(face_position.replace("'", "\""))
        return pos_dict['pitch'], pos_dict['yaw'], pos_dict['roll']
    
    def calculate_attention(pitch, yaw):
        max_angle_pitch = 45  # Assuming that an angle of 45 degrees or more means no attention
        max_angle_yaw = 60
        attention_pitch = max(0, 1 - abs(pitch) / max_angle_pitch)
        attention_yaw = max(0, 1 - abs(yaw) / max_angle_yaw)

        # ponderate the attention by the pitch and yaw
        if attention_pitch == 0 or attention_yaw == 0:
            return 0;

        return (attention_yaw*0.6 + attention_pitch*0.4) * 100

    data[['pitch', 'yaw', 'roll']] = data['Face Position'].apply(lambda x: pd.Series(parse_face_position_json(x)))
    head_pose = data[['student_id', 'Frame', 'pitch', 'yaw', 'roll']]
    head_pose['attention'] = head_pose.apply(lambda row: calculate_attention(row['pitch'], row['yaw']), axis=1)
    

    return head_pose

def attentionOverTime(data):
    head_pose = buildAttentionDataFrame(data)
    # remove student id and group by frame the values are the mean when grouped by frame
    attention_over_time = head_pose.groupby('Frame').mean().reset_index()
    attention_over_time['smoothed_attention'] = attention_over_time['attention'].rolling(window=50).mean()
    # return data as [{name: 00:00:00 (from frame knowing video is at 25fps), attention: 0.0}, ...]
    data = [{'name': f'{int(row["Frame"]/25//60):02d}:{int(row["Frame"]/25%60):02d}:{int(row["Frame"]%25*40):02d}', 'attention': float("{:.2f}".format(row['smoothed_attention']))} for index, row in attention_over_time.iterrows()]
    return data



def radarEmotion(data):
    # emotions count normalized
    emotions_count = data['Main Emotion'].value_counts(normalize=True)
    # save as [{subject: emotion, A: value, fullMark: maxVal}, ...]
    radar_data = []
    fullMark = float("{:.2f}".format(emotions_count.max()))
    for emotion, value in emotions_count.items():
        radar_data.append({'subject': emotion, 'A': float("{:.2f}".format(value)), 'fullMark': fullMark})
    return radar_data

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file_path> <teacher_id>")
        sys.exit(1)
    file_path = sys.argv[1]
    teacher_id = sys.argv[2]
    # Initialize Firebase
    initialize_firebase()
    db = firestore.client()
    # Create the session document
    create_session_document(file_path, teacher_id)
