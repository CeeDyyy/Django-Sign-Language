import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the TM model (We use compile=False because TM models sometimes throw compiler warnings)
base_dir = os.path.dirname(__file__)
tm_model_path = os.path.join(base_dir, 'ai_model', 'keras_model.h5') # Your TM file name!

# Try loading it (wrap in try/except so the server doesn't crash if the file is missing yet)
try:
    tm_model = load_model(tm_model_path, compile=False)
    
    # Load labels from TM's labels.txt
    labels_path = os.path.join(base_dir, 'ai_model', 'labels.txt')
    with open(labels_path, 'r') as f:
        tm_labels = [line.strip().split(' ')[1] for line in f.readlines()]
except:
    tm_model = None
    tm_labels = []
    print("Warning: Teachable Machine model/labels not found yet.")

def process_teachable_machine_frame(frame):
    if tm_model is None:
        return None

    # 1. Resize the image to 224x224 (The exact size TM uses)
    resized_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # 2. Convert to a numpy array and reshape it to (1, 224, 224, 3)
    image_array = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)

    # 3. Normalize the image array (TM math expects pixels between -1 and 1)
    normalized_image_array = (image_array / 127.5) - 1

    # 4. Predict instantly!
    prediction = tm_model.predict(normalized_image_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    if confidence > 0.8: # TM needs a higher confidence threshold to avoid spam
        return tm_labels[index]
    
    return None