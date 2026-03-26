import json
import base64
import numpy as np
import cv2
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from channels.generic.websocket import WebsocketConsumer

# --- 1. SET UP THE BRAIN (Runs once when the server starts) ---
mp_holistic = mp.solutions.holistic
actions = np.array(['hello', 'me', 'you', 'bye']) # Update if you added more!

# Rebuild the neural network structure (Using the tutorial's 1662 shape)
model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))    # Change the input_shape to 258!
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Find and load your saved weights
model_path = os.path.join(os.path.dirname(__file__), 'ai_model', 'action.h5')
model.load_weights(model_path)


# --- 2. MEDIAPIPE HELPER FUNCTIONS ---
def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# def old_extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
#     return np.concatenate([pose, face, lh, rh])
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    # Return only Pose, Left Hand, and Right Hand (132 + 63 + 63 = 258)
    return np.concatenate([pose, lh, rh])


# --- 3. THE LIVE CONSUMER ---
class SignLanguageConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        # Give this specific web user their own empty memory banks
        self.sequence = [] 
        self.predictions = []
        # Turn on the MediaPipe scanner
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("NextJS connected and AI is ready!")

    def disconnect(self, close_code):
        self.holistic.close() # Turn off the scanner when they leave
        print("NextJS disconnected.")

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        frame_data = text_data_json.get('frame')

        if frame_data:
            # 1. Decode Image
            format, imgstr = frame_data.split(';base64,') 
            img_bytes = base64.b64decode(imgstr)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 2. Run MediaPipe Detection
            image, results = mediapipe_detection(frame, self.holistic)
            
            # --- THE HAND GATE ---
            if results.left_hand_landmarks or results.right_hand_landmarks:
                
                # Extract keypoints ONLY if hands are present
                keypoints = extract_keypoints(results)
                
                # 3. Prediction Logic
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                
                if len(self.sequence) == 30:
                    # Add the axis and predict!
                    res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    predicted_word = actions[np.argmax(res)]
                    confidence = float(res[np.argmax(res)])
                    
                    # Using the tutorial's stability check:
                    self.predictions.append(np.argmax(res))
                    if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                        if confidence > 0.5:
                            # SHOOT THE TRANSLATED WORD BACK TO NEXTJS!
                            self.send(text_data=json.dumps({
                                'prediction': predicted_word
                            }))
                            
                            # THE CRITICAL FIX: Wipe the memory! after a successful translation!
                            # This gives the server a cooldown so it doesn't crash.
                            self.sequence = []
                            self.predictions = []
            
            else:
                # NO HANDS DETECTED
                # Wipe the memory so the next sign starts with a clean 30 frames
                self.sequence = []