import cv2
from keras.models import model_from_json
import numpy as np

# Load the model architecture from JSON file
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the trained weights into the model
model.load_weights("emotiondetector.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Dictionary to map predicted indices to emotion labels
labels = {1: 'happy', 2: 'neutral', 3: 'sad'}

# Main loop to capture frames and perform emotion detection
while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region and preprocess it
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img)

        # Make a prediction
        pred = model.predict(img)
        prediction_label = labels[pred.argmax() + 1]  # Adjust index by 1 to match labels

        # Display the predicted emotion label on the frame
        cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
