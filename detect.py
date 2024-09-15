from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/home/mostafabakr8962/Desktop/object_detection/saved_model/object_detection_model.h5", compile=False)

# Load the labels and strip any extra characters like newlines
class_names = [line.strip() for line in open("/home/mostafabakr8962/Desktop/object_detection/saved_model/labels.txt", "r").readlines()]

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("/home/mostafabakr8962/Desktop/object_detection/train/photo-1512941937669-90a1b58e7e9c.jpg").convert("RGB")

# Resize the image to be at least 224x224 and then crop from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print(f"Class: {class_name}")
print(f"Accuracy rate: {confidence_score * 100:.2f}%")
