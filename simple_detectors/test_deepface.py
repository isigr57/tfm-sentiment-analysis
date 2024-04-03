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

# #face verification
# obj = DeepFace.verify(img1_path = "img1.jpg", 
#         img2_path = "img2.jpg", 
#         detector_backend = backends[0]
# )

# #face recognition
# dfs = DeepFace.find(img_path = "img.jpg", 
#         db_path = "my_db", 
#         detector_backend = backends[1]
# )

# #embeddings
# embedding_objs = DeepFace.represent(img_path = "img.jpg", 
#         detector_backend = backends[2]
# )

image_path = './images/emotions.jpeg'

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
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
    
    # Prepare the text with dominant information
    dominant_emotion = result['dominant_emotion']
    age = result['age']
    dominant_gender = result['dominant_gender']
    text = f"{dominant_emotion}, Age: {age}, Gender: {dominant_gender}"
    
    # Paint the text onto the image, slightly above the rectangle
    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

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