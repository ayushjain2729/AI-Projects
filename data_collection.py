import cv2
import numpy as np
import os
import mediapipe as mp

# MediaPipe holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Processes an image using the MediaPipe model to detect landmarks.
    Converts image color space for MediaPipe, processes, then converts back.
    """
    # Convert BGR image to RGB for MediaPipe
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Image is no longer writeable to improve performance
    results = model.process(image) # Make prediction
    image.flags.writeable = True   # Image is now writeable for drawing
    # Convert RGB image back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_separate_bounding_boxes(image, results):
    """
    Draws separate rectangular bounding boxes for the left hand and right hand.
    Face and pose bounding boxes are intentionally excluded.
    """
    h, w, c = image.shape # Get image dimensions
    padding = 10 # Small padding for the bounding boxes

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
    Extracts ONLY hand landmark data into a flattened NumPy array.
    Pose and face landmarks are intentionally excluded from the output.
    If a hand is not detected, it returns an array of zeros of the expected size for that hand.
    """
    # Pose and face landmarks are not extracted, so they contribute zeros of size 0 to concatenation
    pose = np.zeros(0) # Exclude pose landmarks
    face = np.zeros(0) # Exclude face landmarks

    # Extract left hand landmarks (x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3) # 21 landmarks, 3 values each (x,y,z)

    # Extract right hand landmarks (x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3) # 21 landmarks, 3 values each (x,y,z)

    # Concatenate only hand keypoints into a single array
    return np.concatenate([lh, rh]) # Only left hand and right hand keypoints


# --- Configuration Variables ---
# Path for exported data, numpy arrays will be saved here
DATA_PATH = os.path.join('MP_Data')

# Actions (sign language words) that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Number of video sequences to collect for each action
no_sequences = 30

# Number of frames (time steps) in each sequence/video
sequence_length = 30

# --- Setup Data Collection Directories ---
# Create folders for each action and sequence (e.g., MP_Data/hello/0, MP_Data/hello/1, ...)
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            # This error means the directory already exists, which is fine.
            pass
        except Exception as e:
            # Catch any other unexpected errors during directory creation
            print(f"An unexpected error occurred while creating directory {os.path.join(DATA_PATH, action, str(sequence))}: {e}")
            exit() # Exit if directory creation is critical and fails

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0) # 0 refers to the default webcam

# Check if the webcam was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream. Please ensure your webcam is connected and not in use.")
    exit() # Exit the script if webcam cannot be accessed

# --- Graceful Exit Flag ---
# This flag will be used to signal a stop to all nested loops from any point.
stop_collection = False

# --- MediaPipe Holistic Model Initialization ---
# Set mediapipe model with detection and tracking confidence thresholds
# MediaPipe Holistic still detects face and pose internally, but we only extract hand data.
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # --- Main Data Collection Loop ---
    # Loop through each action (e.g., 'hello', 'thanks')
    for action in actions:
        # Check if the stop flag is set from an inner loop or camera error
        if stop_collection:
            break # Break from 'for action' loop

        # Loop through each sequence (video) for the current action
        for sequence in range(no_sequences):
            # Check if the stop flag is set
            if stop_collection:
                break # Break from 'for sequence' loop

            # Loop through each frame within the current sequence
            for frame_num in range(sequence_length):
                # Read feed from webcam
                ret, frame = cap.read()

                # Check if frame was read successfully
                if not ret:
                    print(f"Failed to grab frame for {action} video {sequence}, frame {frame_num}. Stopping data collection.")
                    stop_collection = True # Set flag to stop all loops
                    break # Break from 'for frame_num' loop

                # Make detections using MediaPipe
                image, results = mediapipe_detection(frame, holistic)

                # Draw separate bounding boxes ONLY for hands
                draw_separate_bounding_boxes(image, results)

                # Display collection status messages on the OpenCV feed
                if frame_num == 0:
                    # Message for starting a new collection sequence
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000) # Wait 2 seconds to give user time to prepare for the sign
                else:
                    # Message during frame collection
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Extract keypoints from the MediaPipe results (hands only)
                keypoints = extract_keypoints(results)
                # Define the path to save the keypoints NumPy array
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                # Save the keypoints array
                np.save(npy_path, keypoints)

                # --- Graceful Exit Check ---
                # Check for 'q' key press to quit the entire collection process
                key_pressed = cv2.waitKey(10) & 0xFF # Wait 10ms for a key press
                if key_pressed == ord('q'):
                    stop_collection = True # Set the flag to true
                    break # Break from 'for frame_num' loop (innermost)
                elif key_pressed == 27: # Optional: Allow 'ESC' key to also exit
                    stop_collection = True
                    break

            # Check flag after the innermost loop
            if stop_collection:
                break # Break from 'for sequence' loop

        # Check flag after the middle loop
        if stop_collection:
            break # Break from 'for action' loop

# --- Final Cleanup ---
# Release the webcam resource
cap.release()
# Destroy all OpenCV windows
cv2.destroyAllWindows()

if stop_collection:
    print("\nData collection stopped by user or due to an error.")
else:
    print("\nData collection completed successfully!")