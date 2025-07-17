🤟 Sign Language to English AI Agent 🗣️
An intelligent system that translates American Sign Language (ASL) gestures into English text in real-time, focusing exclusively on hand movements. Built with MediaPipe for robust landmark detection and TensorFlow for deep learning.

✨ Features
Hands-Only Landmark Extraction: Focuses solely on left and right hand keypoints for sign recognition, ignoring face and pose data.

Real-time Gesture Recognition: Translates signs as you perform them in front of your webcam.

Deep Learning Model (LSTM): Utilizes Long Short-Term Memory networks, ideal for sequential data like gestures.

Customizable Actions: Easily extendable to recognize more sign language words.

Visual Feedback: Displays clear bounding boxes around detected hands and the translated text.

Robust Data Collection: Guided process to build your own dataset.

Model Persistence: Saves trained models for later use without retraining.

🚀 How It Works
This project follows a standard machine learning pipeline:

Data Collection (data_collection.py):

Uses Google's MediaPipe Holistic to detect 3D keypoints for your left and right hands.

Captures sequences of these keypoints over time (e.g., 30 frames per sign).

Saves these sequences as NumPy arrays (.npy files) in a structured directory (MP_Data).

Model Training (model_training.py):

Loads the collected hand keypoint sequences.

Preprocesses the data (e.g., one-hot encoding labels).

Builds a Sequential Keras model with multiple LSTM layers to learn temporal patterns in the sign gestures.

Trains the model using the collected data.

Saves the trained model as action.h5.

Real-time Translation (realtime_translation.py):

Loads the pre-trained action.h5 model.

Continuously captures live video from your webcam.

Applies MediaPipe to extract hand keypoints from each frame.

Feeds sequences of live keypoints to the trained LSTM model for prediction.

Applies a smoothing algorithm to stabilize predictions.

Displays the recognized English word on the screen in real-time.

📂 Project Structure
.
├── MP_Data/                      # Directory to store collected hand landmark data
│   ├── hello/                    # Subdirectory for 'hello' sign
│   │   ├── 0/                    # Sequence 0 for 'hello'
│   │   │   ├── 0.npy             # Frame 0 keypoints
│   │   │   └── ...               # ... up to 29.npy
│   │   ├── ...                   # ... up to 29/
│   ├── thanks/                   # Subdirectory for 'thanks' sign
│   │   └── ...
│   └── iloveyou/                 # Subdirectory for 'iloveyou' sign
│       └── ...
├── data_collection.py            # Script for collecting hand landmark data
├── model_training.py             # Script for building and training the LSTM model
├── realtime_translation.py       # Script for real-time sign language translation
└── action.h5                     # (Generated) The trained Keras model file

🛠️ Getting Started
Prerequisites
Before you begin, ensure you have:

Python 3.9+ installed.

pip (Python package installer).

A working webcam.

Installation
Clone the repository:

git clone https://github.com/ayushjain2729/ASL_to_Eng.git
cd ASL_to_Eng.git

Create a virtual environment (recommended):

python -m venv venv

Activate the virtual environment:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install dependencies:

pip install opencv-python numpy mediapipe tensorflow scikit-learn

🚀 Usage Guide
Follow these steps in order to collect data, train your model, and run the real-time translator.

Step 1: Collect Hand Landmark Data
This script will guide you through collecting 30 sequences (videos) for each sign. Ensure your hands are clearly visible in the camera frame.

Important: Delete previous data (if any).
If you previously collected full-body data or want to start fresh, delete the MP_Data folder before running:

rm -rf MP_Data  # For macOS/Linux
rmdir /s /q MP_Data # For Windows

Run the data collection script:

python data_collection.py

A webcam feed window will open.

Follow the on-screen prompts: "STARTING COLLECTION" will give you 2 seconds to prepare.

Perform the sign clearly and consistently for each of the 30 sequences per word.

Press q or ESC to stop collection early.

The script will close automatically once all data is collected.

Step 2: Train the LSTM Model
This script will train your AI model using the collected hand landmark data.

Crucial: Update Model Input Shape!
Since we are now using only hand data, the input shape of the LSTM model must be updated.
Open model_training.py and change the input_shape in the first LSTM layer:

# Find this line:
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, 1662)))
# Change it to:
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, 126)))

(The 126 comes from (21 left hand landmarks * 3 coordinates) + (21 right hand landmarks * 3 coordinates) = 63 + 63 = 126)

Run the model training script:

python model_training.py

You will see output detailing data loading, model summary, and epoch-by-epoch training progress.

This process can take some time depending on your hardware.

Upon successful completion, an action.h5 file will be saved in your project directory.

Step 3: Real-time Sign Language Translation
Now, put your trained model to the test!

Run the real-time translation script:

python realtime_translation.py

A webcam feed window will open.

Perform the signs ('hello', 'thanks', 'iloveyou') you trained the model on.

The recognized English word should appear at the top of the screen.

Press q or ESC to quit the application.

Troubleshooting Webcam Issues (if realtime_translation.py exits immediately)
If the realtime_translation.py script closes immediately, it's almost always a webcam access issue. The script includes enhanced error messages to help you diagnose.

Close other applications: Ensure no other software (Zoom, Teams, browser tabs, other camera apps) is using your webcam.

Restart your computer: This often resolves lingering camera locks.

Check camera connection: For external webcams, ensure it's securely plugged in. Try a different USB port.

Privacy settings: Verify your operating system's camera privacy settings allow Python or your terminal/VS Code to access the webcam.

Windows: Settings > Privacy & security > Camera.

macOS: System Settings > Privacy & Security > Camera.

Camera Index: The script tries indices 0, 1, 2. If you have a unique setup, you might need to manually adjust cv2.VideoCapture(i) in realtime_translation.py.

🔮 Future Enhancements
Expand Vocabulary: Add more signs and corresponding data.

Improved UI: Develop a more user-friendly graphical interface (e.g., using PyQt, Tkinter, or a web framework).

Advanced Smoothing: Implement more sophisticated temporal smoothing algorithms for predictions.

Multi-person Detection: Extend to recognize signs from multiple individuals simultaneously.

Deployment: Explore deploying the model to a web service or mobile application.

🤝 Contributing
Contributions are welcome! If you have suggestions, bug reports, or want to add new features, please feel free to open an issue or submit a pull request.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements
Google MediaPipe: For powerful and easy-to-use pose, face, and hand landmark detection.

TensorFlow & Keras: For the robust deep learning framework.

OpenCV: For camera interaction and image processing.
