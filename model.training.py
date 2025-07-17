import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- Configuration Variables ---
# Path for exported data (should match DATA_PATH in data_collection.py)
DATA_PATH = os.path.join('MP_Data')

# Actions (sign language words) that we are trying to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Number of video sequences collected for each action
no_sequences = 30

# Number of frames (time steps) in each sequence/video
sequence_length = 30

# --- 1. Load Data and Preprocess ---
print("Loading data and preprocessing...")

# Create a mapping from action names to numerical labels
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

# Iterate through each action
for action in actions:
    # Iterate through each sequence (video) for the current action
    for sequence in range(no_sequences):
        window = [] # This will store frames for the current sequence
        # Iterate through each frame within the current sequence
        for frame_num in range(sequence_length):
            # Construct the full path to the .npy file for the current frame
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")

            # Check if the data file exists before attempting to load it
            if os.path.exists(file_path):
                res = np.load(file_path)
                window.append(res)
            else:
                # If a file is missing, print a warning and skip this entire sequence
                print(f"Warning: Missing data file: {file_path}. Skipping sequence {sequence} for action '{action}'.")
                window = [] # Clear incomplete window to prevent partial sequences
                break # Break from inner loop (frame_num), effectively skipping this sequence

        # Only append the sequence and label if the window is complete (i.e., all frames were loaded)
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Skipped incomplete sequence {sequence} for action '{action}' during loading.")

# Convert the list of sequences into a NumPy array
X = np.array(sequences)
# Convert numerical labels to one-hot encoded format (e.g., 0 -> [1,0,0], 1 -> [0,1,0])
y = to_categorical(labels).astype(int) # .astype(int) ensures integer one-hot vectors

# Print the shapes of the loaded data for verification
print(f"Shape of X (sequences): {X.shape}") # Expected: (num_samples, sequence_length, num_keypoints)
print(f"Shape of y (labels): {y.shape}")   # Expected: (num_samples, num_actions)

# Split the data into training and testing sets
# test_size=0.05 means 5% of data for testing, 95% for training
# random_state for reproducibility ensures the same split every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")


# --- 2. Build the LSTM Model ---
print("\nBuilding the LSTM model...")

# Define the directory for TensorBoard logs
log_dir = os.path.join('Logs')
# Create a TensorBoard callback to visualize training progress
tb_callback = TensorBoard(log_dir=log_dir)

# Initialize a Sequential model (layers are added one after another)
model = Sequential()

# First LSTM layer:
# 64 units (neurons), return_sequences=True means output sequence for next LSTM layer
# 'tanh' activation is generally preferred for intermediate LSTM layers over 'relu'
# input_shape: (sequence_length, num_keypoints) -> (30, 1662)
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, 1662)))

# Second LSTM layer:
# 128 units, return_sequences=True
model.add(LSTM(128, return_sequences=True, activation='tanh'))

# Third LSTM layer:
# 64 units, return_sequences=False means output a single vector for the Dense layers
model.add(LSTM(64, return_sequences=False, activation='tanh'))

# Dense (fully connected) layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Output layer:
# Number of units equals the number of actions (classes)
# 'softmax' activation for multi-class classification, outputs probabilities for each class
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
# Optimizer: Adam is a popular choice for deep learning
# Loss function: 'categorical_crossentropy' for multi-class classification with one-hot encoded labels
# Metrics: 'categorical_accuracy' to monitor accuracy during training
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Display the model summary (layers, output shapes, number of parameters)
model.summary()

# --- 3. Train the Model ---
print("\nStarting model training...")
# Train the model using the training data
# epochs: number of times to iterate over the entire dataset
# callbacks: list of callbacks to apply during training (e.g., TensorBoard)
# validation_data: data to evaluate the model on at the end of each epoch to monitor overfitting
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))

# --- 4. Save the Model ---
print("\nSaving the trained model...")
# Save the trained model to a file in Keras H5 format
model.save('action.h5')
print("Model trained and saved as action.h5")

# --- 5. Evaluate the Model (Optional but Recommended) ---
print("\nEvaluating model on test data...")
# Evaluate the model's performance on the unseen test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0) # verbose=0 to suppress progress bar
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")