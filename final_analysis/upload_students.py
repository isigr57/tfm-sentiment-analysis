import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate('serviceAccount.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sapere-83929.appspot.com'
})

db = firestore.client()
bucket = storage.bucket()

# Folder containing your images
image_folder = './faces_db'

# Function to upload image to Firebase Storage and return the URL
def upload_image(image_path):
    blob = bucket.blob(os.path.basename(image_path))
    blob.upload_from_filename(image_path)
   # buil url like https://firebasestorage.googleapis.com/v0/b/sapere-83929.appspot.com/o/0.jpg?alt=media
    return f'https://firebasestorage.googleapis.com/v0/b/sapere-83929.appspot.com/o/{os.path.basename(image_path)}?alt=media'

# Sample student data (replace with your actual data)
students = [
    {"name": "Bob Hsu", "email": "bob@sapere.com", "images": ['0.jpg', '0_iter1.jpg','0_iter1.jpg','0_iter2.jpg','0_iter3.jpg','0_iter4.jpg'], "teachersId": ["aD86mTWOjPcoAHPwDtZ4JTsgEab2"]},
    {"name": "Alice Smith", "email": "alice@sapere.com", "images": ['1.jpg', '1_iter1.jpg','1_iter1.jpg','1_iter2.jpg','1_iter3.jpg','1_iter4.jpg'], "teachersId": ["aD86mTWOjPcoAHPwDtZ4JTsgEab2"]},
    {"name": "Charlie Brown", "email": "charlie@sapere.com", "images": ['2.jpg', '2_iter1.jpg','2_iter1.jpg','2_iter2.jpg','2_iter3.jpg','2_iter4.jpg'], "teachersId": ["aD86mTWOjPcoAHPwDtZ4JTsgEab2"]},
    {"name": "David Johnson", "email": "david@sapere.com", "images": ['3.jpg', '3_iter1.jpg','3_iter1.jpg','3_iter2.jpg','3_iter3.jpg','3_iter4.jpg'], "teachersId": ["aD86mTWOjPcoAHPwDtZ4JTsgEab2"]},
    {"name": "Eva Green", "email": "eva@sapere.com", "images": ['4.jpg', '4_iter1.jpg','4_iter1.jpg','4_iter2.jpg','4_iter3.jpg','4_iter4.jpg'], "teachersId": ["aD86mTWOjPcoAHPwDtZ4JTsgEab2"]}
]

# Process each student
for student in students:
    # Initialize the list of images
    recognition_images = []
    image_path = None

    # Iterate over the files in the folder for the current student
    for filename in student['images']:
        image_path_full = os.path.join(image_folder, filename)
        uploaded_url = upload_image(image_path_full)
        if image_path is None:
            image_path = uploaded_url  # The first image becomes the main imagePath
        else:
            recognition_images.append(uploaded_url)  # Other images go into recognitionImages

    # Create the document data
    student_data = {
        'name': student['name'],
        'email': student['email'],
        'sessions': [],
        'imagePath': image_path,
        'recognitionImages': recognition_images,
        'teachersId': student['teachersId'],
        'createdAt': datetime.now()
    }

    # Add or update the student document in Firestore
    db.collection('students').add(student_data)

print("Students documents created/updated successfully.")
