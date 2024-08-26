
import pandas as pd
import json
from datetime import datetime
import sys
import matplotlib.pyplot as plt


def create_session_analysis(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Extract unique student_ids
    student_ids = data['student_id'].unique()
    
    # Create a new data frame with only frames from 0 to max frame every 60 frames
    new_data = pd.DataFrame({'Frame': range(0, data['Frame'].max(), 60)})
    
    # Merge the new data frame with the original data frame to get the presence column
    data = pd.merge(new_data, data, on='Frame', how='left')
    
    # Create references for students
    student_references = [f'{student_id}' for student_id in student_ids]
    
    # Extract student data
    students_data = extractStudentsData(student_references)
    
    # Build attention data frame
    attDf = buildAttentionDataFrame(data)

    # Print analysis results
    print("Main Emotion: ", extractMainEmotion(data))
    print("Overall Attention: ", float("{:.1f}".format(attDf['attention'][attDf['attention'] != 0].mean())))

    # Plot overall attention over time
    attention_data = attentionOverTime(data)
    plt.figure(figsize=(15, 7))
    plt.plot([d['name'] for d in attention_data], [d['attention'] for d in attention_data], label='Overall Attention')
    plt.ylabel('Attention Level')
    plt.xlabel('Time')
    plt.xticks([])
    plt.title('Overall Attention Over Time')
    plt.legend()
    plt.savefig('overall_attention.png')

    # Plot emotions over time
    emotions_data = emotionsOverTime(data)
    plt.figure(figsize=(15, 7))
    
    for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
        plt.plot([d['name'] for d in emotions_data], [d[emotion] for d in emotions_data], label=emotion.capitalize())
    plt.xlabel('Time')
    plt.xticks([])
    plt.ylabel('Emotion Probability')
    plt.title('Emotions Over Time')
    plt.legend()
    plt.savefig('emotions_over_time.png')
    

    # Radar plot for emotion distribution (requires a separate plotting function)
    radar_data = radarEmotion(data)
    plot_radar_chart(radar_data)

    for student_id, student_data in students_data.items():
        print(f"Student {student_id}:")
        print(f"Main Emotion: {student_data['mainEmotion']}")
        print(f"Overall Attention: {student_data['overallAttention']}")
        
        plt.figure(figsize=(15, 7))
        plt.plot([d['name'] for d in student_data['attentionOverTime']], [d['attention'] for d in student_data['attentionOverTime']], label='Attention')
        plt.ylabel('Attention Level')
        plt.xlabel('Time')
        plt.xticks([])
        plt.title(f'Attention Over Time for Student {student_id}')
        plt.legend()
        plt.savefig(f'attention_{student_id}.png')

        plt.figure(figsize=(15, 7))
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
            plt.plot([d['name'] for d in student_data['emotionsOverTime']], [d[emotion] for d in student_data['emotionsOverTime']], label=emotion.capitalize())
        plt.xlabel('Time')
        plt.xticks([])
        plt.ylabel('Emotion Probability')
        plt.title(f'Emotions Over Time for Student {student_id}')
        plt.legend()
        plt.savefig(f'emotions_{student_id}.png')

        plot_radar_chart(student_data['emotionRadar'])

        plt.figure(figsize=(15, 7))
        plt.plot([d['name'] for d in student_data['presenceOverTime']], [d['presence'] for d in student_data['presenceOverTime']], label='Presence')
        plt.ylabel('Presence')
        plt.xlabel('Time')
        plt.xticks([])
        plt.title(f'Presence Over Time for Student {student_id}')
        plt.legend()
        plt.savefig(f'presence_{student_id}.png')





def plot_radar_chart(radar_data, student_id=None):
    # Radar chart plotting function
    categories = [d['subject'] for d in radar_data]
    values = [d['A'] for d in radar_data]
    max_val = max([d['fullMark'] for d in radar_data])

    angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
    values += values[:1]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories)

    ax.plot(angles, values)
    ax.fill(angles, values, 'b', alpha=0.1)

    ax.set_ylim(0, max_val)

    if student_id:
        plt.title(f'Emotion Radar for Student {student_id}')
    else:
        plt.title('Emotion Radar')
    
    plt.savefig('emotion_radar.png')

def extractStudentsData(student_references):
    data = pd.read_csv(file_path)
    students_data = {}
    for student in student_references:
        filtered_data = data[data['student_id'] == student]
        # create a new data frame with only frame fro 0 to max frame every 60 frames
        new_data = pd.DataFrame({'Frame': range(0, data['Frame'].max(), 60)})
        # merge the new data frame with the original data frame to get the presence column
        filtered_data = pd.merge(new_data, filtered_data, on='Frame', how='left')
        attDf = buildAttentionDataFrame(filtered_data)
        students_data[student] = {
            'mainEmotion': extractMainEmotion(filtered_data),
            'overallAttention': float("{:.1f}".format(attDf['attention'][attDf['attention'] != 0].mean())),
            'emotionRadar': radarEmotion(filtered_data),
            'attentionOverTime': attentionOverTime(filtered_data),
            'presenceOverTime': presenceOverTime(filtered_data),
            'emotionsOverTime': emotionsOverTime(filtered_data)
        }
    return students_data

def presenceOverTime(data):
    data['presence'] = data['student_id'].notnull().astype(int)
    data = data[['Frame', 'presence']]
    data = data.groupby('Frame').mean().reset_index()
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
    data = data[['Frame', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']]
    data = data.groupby('Frame').mean().reset_index()
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
    attention_over_time = head_pose.groupby('Frame').mean().reset_index()

    attention_over_time['smoothed_attention'] = attention_over_time['attention'].rolling(window=50).mean()

    data = [{'name': f'{int(row["Frame"]/25//60):02d}:{int(row["Frame"]/25%60):02d}:{int(row["Frame"]%25*40):02d}', 
             'attention': float("{:.2f}".format(row['smoothed_attention']))} for index, row in attention_over_time.iterrows()]
             
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
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    # Create the session document
    create_session_analysis(file_path)
