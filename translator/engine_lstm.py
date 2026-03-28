import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- AI SETUP ---
mp_holistic = mp.solutions.holistic
actions = np.array(['hello', 'me', 'you', 'bye'])

# 1. Create a machine that builds our skeletons
def build_sign_language_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model

# 2. Build the two skeletons with different shapes
model_default = build_sign_language_model((30, 1662))
model_less_face = build_sign_language_model((30, 258))

# 3. Find the weights files
base_dir = os.path.dirname(__file__)
default_path = os.path.join(base_dir, 'ai_model', 'action(4Default).h5')
less_face_path = os.path.join(base_dir, 'ai_model', 'action(4FaceLess).h5')

# 4. Load the weights onto the skeletons!
model_default.load_weights(default_path)
model_less_face.load_weights(less_face_path)

# --- 2. MEDIAPIPE HELPER FUNCTIONS ---
def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results, model_type='default'):
    # Always extract these three (Pose: 132, LH: 63, RH: 63)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Switchboard Logic!
    if model_type == 'default':
        # Add the 1404 face landmarks
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        return np.concatenate([pose, face, lh, rh]) # Returns exactly 1662

    elif model_type == 'less_face':
        # Skip the face entirely
        return np.concatenate([pose, lh, rh]) # Returns exactly 258

# --- THE ENGINE CLASS ---
class LSTMEngine:
    def __init__(self):
        self.sequence = []
        self.predictions = []
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(self, frame, model_type):
        # 1. MediaPipe Detection
        image, results = mediapipe_detection(frame, self.holistic)
        results = self.holistic.process(image)
        
        # 2. Hand Gate
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results, model_type)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]

            if len(self.sequence) == 30:
                # Pick the brain
                active_model = model_less_face if model_type == 'less_face' else model_default
                
                # Predict
                res = active_model.predict(np.expand_dims(self.sequence, axis=0))[0]
                predicted_word = actions[np.argmax(res)]
                confidence = float(res[np.argmax(res)])

                self.predictions.append(np.argmax(res))
                
                # Stability Check
                if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                    if confidence > 0.5:
                        self.sequence = [] # Wipe memory
                        self.predictions = []
                        return predicted_word # Return the word!
        else:
            # Wipe memory if hands drop
            self.sequence = []
            self.predictions = []
            
        return None # Return nothing if no sign is finished