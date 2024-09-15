from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/home/mostafabakr8962/Desktop/object_detection/saved_model/object_detection_model.h5", compile=False)

# Load the labels
class_names = open("/home/mostafabakr8962/Desktop/object_detection/saved_model/labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

while True:
    # Grab the webcam's image
    ret, image = camera.read()

    # Resize the image to 1280x720
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)

    # Create detection area in the center (rectangle)
    detection_area_start = (300, 180)  # (x1, y1)
    detection_area_end = (1000, 650)    # (x2, y2)
    cv2.rectangle(image, detection_area_start, detection_area_end, (0, 255, 0), 2)

    # Extract the region of interest (ROI) for object detection (inside the rectangle)
    roi = image[180:540, 440:840]

    # Resize the ROI to match the model's input size (224x224)
    roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)

    # Preprocess the ROI for the model
    roi_resized = np.asarray(roi_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    roi_resized = (roi_resized / 127.5) - 1  # Normalize

    # Predict with the model
    prediction = model.predict(roi_resized)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove any newlines
    confidence_score = prediction[0][index]

    # Display class name and confidence score on the screen
    text = f"Class: {class_name} | Confidence: {str(np.round(confidence_score * 100))[:-2]}%"
    
    # Adjust the text position, font size, and color for better visibility
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # Thicker text for visibility

    # Show the image with detection area in a window
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses (ESC to quit)
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
