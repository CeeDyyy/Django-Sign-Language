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
# 1. Create a machine that builds our skeletons
def build_sign_language_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax')) # Make sure 'actions' is defined above this!
    return model

# 2. Build the two skeletons with different shapes
model_default = build_sign_language_model((30, 1662))
model_less_face = build_sign_language_model((30, 258))

# Find and load your saved weights
# model_path = os.path.join(os.path.dirname(__file__), 'ai_model', 'action.h5')
# model.load_weights(model_path)

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


# --- 3. THE LIVE CONSUMER ---
class SignLanguageConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        # Give this specific web user their own empty memory banks
        self.sequence = [] 
        self.predictions = []
        # Turn on the MediaPipe scanner
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # 2. SET THE DEFAULT ACTIVE BRAIN
        self.active_model = model_default
        print("NextJS connected and AI is ready!")

    def disconnect(self, close_code):
        self.holistic.close() # Turn off the scanner when they leave
        print("NextJS disconnected.")

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        
        # --- 3. THE COMMAND LISTENER ---
        command = text_data_json.get('command')
        if command == 'switch_model':
            selected_model = text_data_json.get('model')
            
            if selected_model == 'less_face':
                self.active_model = model_less_face
                print("Switched to Less Face model.")
            else:
                self.active_model = model_default
                print("Switched to Default model.")
                
            # Wipe memory so we don't mix data between models!
            self.sequence = []
            self.predictions = []
            return # Skip the rest of the function for this message

        # --- NORMAL VIDEO PROCESSING ---
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
                
                # 1. Figure out which model is currently active
                current_type = 'less_face' if self.active_model == model_less_face else 'default'
                
                # 2. Tell the extractor which shape to build!
                # Extract keypoints ONLY if hands are present
                # keypoints = extract_keypoints(results)
                keypoints = extract_keypoints(results, model_type=current_type)
                
                # 3. Prediction Logic
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                
                if len(self.sequence) == 30:
                    # 4. PREDICT WITH THE ACTIVE BRAIN!
                    # Add the axis and predict!
                    res = self.active_model.predict(np.expand_dims(self.sequence, axis=0))[0]
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
                self.predictions = [] # <-- Add this line for a perfect clean slate!