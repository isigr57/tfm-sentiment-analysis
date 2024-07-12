import cv2
import dlib
import numpy as np
from deepface import DeepFace

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Create a list to store trackers and a dictionary to map trackers to their IDs
trackers = []
tracker_labels = {}

# Function to generate a random color
def get_random_color():
    return (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))

# Function to create a new tracker for a face
def create_tracker(frame, rect, tracker_id):
    tracker = dlib.correlation_tracker()
    tracker.start_track(frame, rect)
    trackers.append((tracker, get_random_color(), tracker_id))

# Update and draw rectangles for all trackers
def update_trackers(frame, rgb_frame):
    for tracker, color, tracker_id in trackers:
        tracker.update(rgb_frame)
        pos = tracker.get_position()
        start_point = (int(pos.left()), int(pos.top()))
        end_point = (int(pos.right()), int(pos.bottom()))

        # Crop the face from the frame for emotion analysis
        face_image = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        
        if face_image.size != 0:
            # Convert cropped face to the color scheme expected by deepface
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            emotion = analyze_emotion(face_image_rgb)

            # Draw rectangle and emotion text
            cv2.rectangle(frame, start_point, end_point, color, 2)
            cv2.putText(frame, f"Person_{tracker_id}: {emotion}", (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA.left(), boxB.left())
    yA = max(boxA.top(), boxB.top())
    xB = min(boxA.right(), boxB.right())
    yB = min(boxA.bottom(), boxB.bottom())

    # Compute the area of intersection
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA.right() - boxA.left() + 1) * (boxA.bottom() - boxA.top() + 1)
    boxBArea = (boxB.right() - boxB.left() + 1) * (boxB.bottom() - boxB.top() + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection / float(boxAArea + boxBArea - intersection)

    # Return the intersection over union value
    return iou

from deepface import DeepFace

def analyze_emotion(face_image):
    try:
        # Analyzing the emotion of the cropped face image
        analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        
        # Since the analysis result is a list, get the first item
        if isinstance(analysis, list) and len(analysis) > 0:
            analysis_dict = analysis[0]
            
            if "emotion" in analysis_dict and "dominant_emotion" in analysis_dict:
                # Get the dominant emotion from the analysis result
                dominant_emotion = analysis_dict["dominant_emotion"]
                return dominant_emotion
            else:
                print("Emotion or dominant_emotion key not found in analysis.")
                return "Unknown"
        else:
            print("Analysis result is not in expected format or empty.")
            return "Unknown"
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return "Unknown"


# Detect and track faces
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    current_tracker_id = 1
    frame_count = 0  # Initialize frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Increment frame_count
        frame_count += 1

        # Check for new faces periodically
        if len(trackers) == 0 or frame_count % 60 == 0:  # Check every 60 frames
            faces = detector(rgb_frame, 0)
            for face in faces:
                # Convert dlib rectangle to a form compatible with the calculate_iou function
                dlib_rect = dlib.rectangle(face.left(), face.top(), face.right(), face.bottom())

                # Check if face is already being tracked
                already_tracked = any(calculate_iou(dlib_rect, dlib.rectangle(int(tracker[0].get_position().left()),
                                                                              int(tracker[0].get_position().top()),
                                                                              int(tracker[0].get_position().right()),
                                                                              int(tracker[0].get_position().bottom()))) > 0.6 for tracker in trackers)
                if not already_tracked:
                    create_tracker(rgb_frame, face, current_tracker_id)
                    current_tracker_id += 1

        # Update all trackers and draw rectangles
        update_trackers(frame, rgb_frame)
        
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
process_video('./videos/test4.mp4')
