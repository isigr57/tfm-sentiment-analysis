import firebase_admin
from firebase_admin import credentials, firestore, storage
import pandas as pd
import json
from datetime import datetime
import sys
import matplotlib.pyplot as plt

def initialize_firebase():
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate('./serviceAccount.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'sapere-83929.appspot.com'
    })
    print("Firebase initialized successfully.")

def create_session_document(file_path, video_path, teacher_id):
    # Load the CSV file
    data = pd.read_csv(file_path)
    # Extract unique student_ids
    student_ids = data['student_id'].unique()
    # The student must have data every 60 frames so if in frame 60 the is a row for the student then the student is present if not the student is absent in that frame 
    # if the student is absent in one frame we will add a row with the student id
    # create a new data frame with only frame fro 0 to max frame every 60 frames
    new_data = pd.DataFrame({'Frame': range(0, data['Frame'].max(), 60)})
    # merge the new data frame with the original data frame to get the presence column
    data = pd.merge(new_data, data, on='Frame', how='left')
    # Create references for students
    student_references = [f'{student_id}' for student_id in student_ids]
    # Create the session document
    session_doc = {
        'name': "Class",
        'createdAt': datetime.now(),
        'students': student_references,
        'teacherId': teacher_id,
        'videoUrl': upload_video_to_storage(video_path),
        'sessionData': {
            'mainEmotion': extractMainEmotion(data),
            'overallAttention': float("{:.1f}".format(buildAttentionDataFrame(data)['attention'].mean())),
            'emotionRadar': radarEmotion(data),
            'attentionOverTime': attentionOverTime(data)
        },
        'studentsData': extractStudentsData(student_references)
    }
    # Add the session document to Firestore
    newSession = db.collection('sessions').add(session_doc)

    for student in student_references:
        doc = db.collection('students').document(student)
        doc.update({
            'sessions': firestore.ArrayUnion([newSession[1].id])
        })
    print("Session document created successfully.")

def extractStudentsData(student_references):
    data = pd.read_csv(file_path)
    students_data = {}
    for student in student_references:
        filtered_data = data[data['student_id'] == student]
        # create a new data frame with only frame fro 0 to max frame every 60 frames
        new_data = pd.DataFrame({'Frame': range(0, data['Frame'].max(), 60)})
        # merge the new data frame with the original data frame to get the presence column
        filtered_data = pd.merge(new_data, filtered_data, on='Frame', how='left')
        students_data[student] = {
            'mainEmotion': extractMainEmotion(filtered_data),
            'overallAttention': float("{:.1f}".format(buildAttentionDataFrame(filtered_data)['attention'].mean())),
            'emotionRadar': radarEmotion(filtered_data),
            'attentionOverTime': attentionOverTime(filtered_data),
            'presenceOverTime': presenceOverTime(filtered_data),
            'emotionsOverTime': emotionsOverTime(filtered_data)
        }
    return students_data

def presenceOverTime(data):
    data['presence'] = data['student_id'].notnull().astype(int)
    # return data as [{name: 00:00:00 (from frame knowing video is at 25fps), presence: 0}, ...]
    data = [{'name': f'{int(row["Frame"]/25//60):02d}:{int(row["Frame"]/25%60):02d}:{int(row["Frame"]%25*40):02d}', 'presence': int(row['presence'])} for index, row in data.iterrows()]
    return data

def emotionsOverTime(data):
    def parse_emotions_json(emotions):
        try:
            emotions_dict = json.loads(emotions.replace("'", "\""))
            return emotions_dict['angry'], emotions_dict['disgust'], emotions_dict['fear'], emotions_dict['happy'], emotions_dict['sad'], emotions_dict['surprise'], emotions_dict['neutral']
        except:
            return 0, 0, 0, 0, 0, 0, 1
    
    # Parse the emotions column and create new columns for each emotion maintaining the frame
    data[['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']] = data['Emotions'].apply(lambda x: pd.Series(parse_emotions_json(x)))
    # Group by frame and calculate the mean for each emotion
    # smmothed values are calculated by rolling mean of 50 frames
    data['smoothed_angry'] = data['angry'].rolling(window=50).mean()
    data['smoothed_disgust'] = data['disgust'].rolling(window=50).mean()
    data['smoothed_fear'] = data['fear'].rolling(window=50).mean()
    data['smoothed_happy'] = data['happy'].rolling(window=50).mean()
    data['smoothed_sad'] = data['sad'].rolling(window=50).mean()
    data['smoothed_surprise'] = data['surprise'].rolling(window=50).mean()
    data['smoothed_neutral'] = data['neutral'].rolling(window=50).mean()


    # return data as [{name: 00:00:00 (from frame knowing video is at 25fps), angry: 0.0, disgust: 0.0, fear: 0.0, happy: 0.0, sad: 0.0, surprise: 0.0, neutral: 0.0}, ...]
    data = [{'name': f'{int(row["Frame"]/25//60):02d}:{int(row["Frame"]/25%60):02d}:{int(row["Frame"]%25*40):02d}', 'angry': float("{:.2f}".format(row['smoothed_angry'])), 'disgust': float("{:.2f}".format(row['smoothed_disgust'])), 'fear': float("{:.2f}".format(row['smoothed_fear'])), 'happy': float("{:.2f}".format(row['smoothed_happy'])), 'sad': float("{:.2f}".format(row['smoothed_sad'])), 'surprise': float("{:.2f}".format(row['smoothed_surprise'])), 'neutral': float("{:.2f}".format(row['smoothed_neutral']))} for index, row in data.iterrows()]
    return data;

def upload_video_to_storage(video_path):

    bucket = storage.bucket()
    gs_path = video_path.split("/")[-1]
    blob = bucket.blob(gs_path)
    blob.upload_from_filename(video_path)
    return f'https://firebasestorage.googleapis.com/v0/b/sapere-83929.appspot.com/o/{gs_path}?alt=media'

def extractMainEmotion(data):
    emotions_count = data['Main Emotion'].value_counts()
    return emotions_count.idxmax()

def buildAttentionDataFrame(data):
    # Function to parse the Face Position column
    def parse_face_position_json(face_position):
        try:
            pos_dict = json.loads(face_position.replace("'", "\""))
            return pos_dict['pitch'], pos_dict['yaw'], pos_dict['roll']
        except:
            return 90, 90, 90

    
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
    if len(sys.argv) != 4:
        print("Usage: python script.py <file_path> <video_path> <teacher_id> ")
        sys.exit(1)
    file_path = sys.argv[1]
    video_path = sys.argv[2]
    teacher_id = sys.argv[3]
    # Initialize Firebase
    initialize_firebase()
    db = firestore.client()
    # Create the session document
    create_session_document(file_path, video_path, teacher_id)
