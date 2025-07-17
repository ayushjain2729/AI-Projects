import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import Counter # Import Counter for robust smoothing logic

# --- Re-used Utility Functions (Ideally, these would be in a separate 'utils.py' file) ---

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Processes an image using the MediaPipe model to detect landmarks.
    Converts image color space for MediaPipe, processes, then converts back.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR image to RGB for MediaPipe
    image.flags.writeable = False                  # Image is no longer writeable to improve performance
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable for drawing
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert RGB image back to BGR for OpenCV display
    return image, results

def draw_separate_bounding_boxes(image, results):
    """
    Draws separate rectangular bounding boxes for the face, left hand, and right hand.
    Each box will have a distinct color.
    """
    h, w, c = image.shape # Get image dimensions
    padding = 10 # Small padding for the bounding boxes

    # --- Draw Face Bounding Box ---
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        # Get min/max x, y coordinates for face landmarks
        face_x = [int(lm.x * w) for lm in face_landmarks]
        face_y = [int(lm.y * h) for lm in face_landmarks]
        min_face_x, max_face_x = min(face_x), max(face_x)
        min_face_y, max_face_y = min(face_y), max(face_y)

        # Apply padding and ensure coordinates are within image bounds
        min_face_x = max(0, min_face_x - padding)
        min_face_y = max(0, min_face_y - padding)
        max_face_x = min(w, max_face_x + padding)
        max_face_y = min(h, max_face_y + padding)

        # Draw green rectangle for face
        cv2.rectangle(image, (min_face_x, min_face_y), (max_face_x, max_face_y), (0, 255, 0), 2)

    # --- Draw Left Hand Bounding Box ---
    if results.left_hand_landmarks:
        left_hand_landmarks = results.left_hand_landmarks.landmark
        # Get min/max x, y coordinates for left hand landmarks
        lh_x = [int(lm.x * w) for lm in left_hand_landmarks]
        lh_y = [int(lm.y * h) for lm in left_hand_landmarks]
        min_lh_x, max_lh_x = min(lh_x), max(lh_x)
        min_lh_y, max_lh_y = min(lh_y), max(lh_y)

        # Apply padding and ensure coordinates are within image bounds
        min_lh_x = max(0, min_lh_x - padding)
        min_lh_y = max(0, min_lh_y - padding)
        max_lh_x = min(w, max_lh_x + padding)
        max_lh_y = min(h, max_lh_y + padding)

        # Draw blue rectangle for left hand
        cv2.rectangle(image, (min_lh_x, min_lh_y), (max_lh_x, max_lh_y), (255, 0, 0), 2)

    # --- Draw Right Hand Bounding Box ---
    if results.right_hand_landmarks:
        right_hand_landmarks = results.right_hand_landmarks.landmark
        # Get min/max x, y coordinates for right hand landmarks
        rh_x = [int(lm.x * w) for lm in right_hand_landmarks]
        rh_y = [int(lm.y * h) for lm in right_hand_landmarks]
        min_rh_x, max_rh_x = min(rh_x), max(rh_x)
        min_rh_y, max_rh_y = min(rh_y), max(rh_y)

        # Apply padding and ensure coordinates are within image bounds
        min_rh_x = max(0, min_rh_x - padding)
        min_rh_y = max(0, min_rh_y - padding)
        max_rh_x = min(w, max_rh_x + padding)
        max_rh_y = min(h, max_rh_y + padding)

        # Draw red rectangle for right hand
        cv2.rectangle(image, (min_rh_x, min_rh_y), (max_rh_x, max_rh_y), (0, 0, 255), 2)


def extract_keypoints(results):
    """
    Extracts landmark data (pose, face, left hand, right hand) into a flattened NumPy array.
    If a landmark type is not detected, it returns an array of zeros of the expected size.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# --- Main Script Logic ---

# Load the trained model
try:
    model = load_model('action.h5')
    print("Model 'action.h5' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'action.h5' exists in the same directory as this script after training.")
    exit() # Exit the script if the model cannot be loaded

# --- Real-Time Prediction Logic Variables ---
sequence = []        # Stores the last `sequence_length` frames of keypoints for prediction
sentence = []        # Stores the recognized words to form a sentence
predictions = []     # Stores raw prediction indices for smoothing
threshold = 0.8      # Confidence threshold: prediction must be above this to be accepted

# Actions array MUST match the order used during training
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence_length = 30 # Must match the sequence_length used during data collection and training
smoothing_window_size = 10 # Number of recent predictions to consider for smoothing


# --- Initialize Webcam ---
cap = cv2.VideoCapture(0) # 0 refers to the default webcam

# Check if the webcam was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream. Please ensure your webcam is connected and not in use.")
    exit() # Exit the script if webcam cannot be accessed

# --- MediaPipe Holistic Model Initialization ---
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    print("Starting real-time translation. Press 'q' or 'ESC' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()

        # Check if frame was read successfully
        if not ret:
            print("Failed to grab frame. Exiting...")
            break # Exit the loop if no frame is received

        # Make detections using MediaPipe
        image, results = mediapipe_detection(frame, holistic)

        # Draw separate bounding boxes for face and hands
        draw_separate_bounding_boxes(image, results)

        # Extract keypoints from the current frame
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        # Keep only the last `sequence_length` frames in the sequence list
        sequence = sequence[-sequence_length:]

        # Perform prediction only when the sequence buffer is full
        if len(sequence) == sequence_length:
            # Expand dimensions to match model input shape (1, sequence_length, num_keypoints)
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0] # verbose=0 suppresses Keras output
            predictions.append(np.argmax(res)) # Store the index of the highest probability prediction

            # Keep only the last `smoothing_window_size` predictions for smoothing
            predictions = predictions[-smoothing_window_size:]

            # --- Prediction Smoothing Logic ---
            # Only apply smoothing if we have enough predictions in the window
            if len(predictions) == smoothing_window_size:
                # Find the most common prediction index in the smoothing window
                from collections import Counter
                most_common_prediction_idx = Counter(predictions).most_common(1)[0][0]

                # Check if the most common prediction's probability from the *current* frame
                # is above the confidence threshold
                if res[most_common_prediction_idx] > threshold:
                    current_action = actions[most_common_prediction_idx]

                    # Append the recognized action to the sentence if it's new or the first word
                    if len(sentence) > 0:
                        if current_action != sentence[-1]: # Avoid adding duplicate consecutive words
                            sentence.append(current_action)
                    else:
                        sentence.append(current_action)

            # Limit the displayed sentence length to show only recent words
            if len(sentence) > 5:
                sentence = sentence[-5:] # Show only the last 5 words

        # --- Display Prediction on Screen ---
        # Draw a black rectangle at the top of the frame for the text background
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1) # Orange background
        # Put the predicted sentence text on the image
        cv2.putText(image, ' '.join(sentence), (3,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # White text

        # Show the processed frame
        cv2.imshow('OpenCV Feed', image)

        # --- Graceful Exit ---
        # Check for 'q' or 'ESC' key press to quit
        key_pressed = cv2.waitKey(10) & 0xFF
        if key_pressed == ord('q') or key_pressed == 27: # 27 is the ASCII for ESC key
            break

# --- Final Cleanup ---
cap.release()        # Release the webcam resource
cv2.destroyAllWindows() # Close all OpenCV windows
print("\nReal-time translation stopped.")