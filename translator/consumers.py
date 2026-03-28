import json
import base64
import numpy as np
import cv2
from channels.generic.websocket import WebsocketConsumer

# Import the two AI engines we just created!
from .engine_lstm import LSTMEngine
from .engine_tm import process_teachable_machine_frame

class SignLanguageConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        # Initialize the LSTM engine so it can remember the 30 frames
        self.lstm_engine = LSTMEngine() 
        self.active_model = 'default' # Start with Default LSTM
        print("NextJS connected and Router is ready!")

    def disconnect(self, close_code):
        self.lstm_engine.holistic.close()
        print("NextJS disconnected.")

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        
        # --- 1. THE COMMAND LISTENER ---
        command = text_data_json.get('command')
        if command == 'switch_model':
            self.active_model = text_data_json.get('model')
            print(f"Router switched to: {self.active_model}")
            
            # Wipe the LSTM memory just in case we are switching away from it
            self.lstm_engine.sequence = []
            self.lstm_engine.predictions = []
            return 

        # --- 2. THE VIDEO ROUTER ---
        frame_data = text_data_json.get('frame')
        if frame_data:
            # Decode Image
            format, imgstr = frame_data.split(';base64,') 
            img_bytes = base64.b64decode(imgstr)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            predicted_word = None

            # Route the traffic based on the dropdown!
            if self.active_model in ['default', 'less_face']:
                # Send to engine_lstm.py
                predicted_word = self.lstm_engine.process_frame(frame, self.active_model)
                
            elif self.active_model == 'teachable_machine':
                # Send to engine_tm.py
                predicted_word = process_teachable_machine_frame(frame)

            # --- 3. SEND RESULT TO NEXTJS ---
            if predicted_word:
                self.send(text_data=json.dumps({
                    'prediction': predicted_word
                }))