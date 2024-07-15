import cv2
from fer_pytorch.fer import FER

fer = FER()
print(fer.device)

fer.get_pretrained_model(model_name="resnet34")

# Test multi
frame = cv2.imread("images/emotions.jpeg")
result = fer.predict_image(frame, show_top=True, path_to_output="./multi.jpg")
print(result)
