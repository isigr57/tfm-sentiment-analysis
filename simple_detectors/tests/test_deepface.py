from deepface import DeepFace
import cv2

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

image_path = '../../images/emotions.jpeg'

#facial analysis
analysis_results = DeepFace.analyze(img_path = image_path, 
        detector_backend = backends[4]
)

img = cv2.imread(image_path)

# Iterate through each analysis result and draw rectangles and text
for result in analysis_results:
    # Get the region of the detected face
    region = result['region']
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    
    # Draw rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Prepare the text with dominant information
    dominant_emotion = result['dominant_emotion']
    age = result['age']
    dominant_gender = result['dominant_gender']
    
    # Paint the text onto the image, slightly above the rectangle
    cv2.putText(img, f"Emotion: {dominant_emotion}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, 2)
    cv2.putText(img, f"Age: {age}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, 2)
    cv2.putText(img, f"Gender: {dominant_gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, 2)


# Save or display the modified image
output_path = 'output.jpg'
cv2.imwrite(output_path, img)
# To display the image, uncomment the lines below
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# #face detection and alignment
# face_objs = DeepFace.extract_faces(img_path = "img.jpg", 
#         target_size = (224, 224), 
#         detector_backend = backends[4]
# )