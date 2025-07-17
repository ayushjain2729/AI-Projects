<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body class="p-4 sm:p-6 md:p-8">
    <div class="container mx-auto bg-white p-6 sm:p-8 md:p-10 rounded-lg shadow-xl">
        <h1 class="text-4xl sm:text-5xl font-extrabold text-center mb-6 flex items-center justify-center">
            <span class="mr-3">🤟</span> Sign Language to English AI Agent <span class="ml-3">🗣️</span>
        </h1>

  <p class="text-lg text-gray-700 mb-8 text-center">
            An intelligent system that translates American Sign Language (ASL) gestures into English text in real-time, focusing exclusively on hand movements. Built with MediaPipe for robust landmark detection and TensorFlow for deep learning.
        </p>
   <!-- Badges Section -->
        <div class="flex flex-wrap justify-center gap-4 mb-10">
            <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python 3.9+ Badge">
            <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow" alt="TensorFlow 2.x Badge">
            <img src="https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras" alt="Keras 2.x Badge">
            <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv" alt="OpenCV 4.x Badge">
            <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT Badge">
        </div>

 <h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">✨ Features</h2>
        <ul class="list-disc pl-6 mb-8 text-gray-700 space-y-2">
            <li><strong>Hands-Only Landmark Extraction:</strong> Focuses solely on left and right hand keypoints for sign recognition, ignoring face and pose data.</li>
            <li><strong>Real-time Gesture Recognition:</strong> Translates signs as you perform them in front of your webcam.</li>
            <li><strong>Deep Learning Model (LSTM):</strong> Utilizes Long Short-Term Memory networks, ideal for sequential data like gestures.</li>
            <li><strong>Customizable Actions:</strong> Easily extendable to recognize more sign language words.</li>
            <li><strong>Visual Feedback:</strong> Displays clear bounding boxes around detected hands and the translated text.</li>
            <li><strong>Robust Data Collection:</strong> Guided process to build your own dataset.</li>
            <li><strong>Model Persistence:</strong> Saves trained models for later use without retraining.</li>
        </ul>

 <h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">🚀 How It Works</h2>
        <p class="text-gray-700 mb-4">This project follows a standard machine learning pipeline:</p>
        <ol class="list-decimal pl-6 mb-8 text-gray-700 space-y-4">
            <li>
                <strong>Data Collection (<code class="text-sm font-semibold">data_collection.py</code>):</strong>
                <ul class="list-disc pl-6 mt-2 space-y-1">
                    <li>Uses Google's <strong>MediaPipe Holistic</strong> to detect 3D keypoints for your left and right hands.</li>
                    <li>Captures sequences of these keypoints over time (e.g., 30 frames per sign).</li>
                    <li>Saves these sequences as NumPy arrays (<code class="text-sm font-semibold">.npy</code> files) in a structured directory (<code class="text-sm font-semibold">MP_Data</code>).</li>
                </ul>
            </li>
            <li>
                <strong>Model Training (<code class="text-sm font-semibold">model_training.py</code>):</strong>
                <ul class="list-disc pl-6 mt-2 space-y-1">
                    <li>Loads the collected hand keypoint sequences.</li>
                    <li>Preprocesses the data (e.g., one-hot encoding labels).</li>
                    <li>Builds a <strong>Sequential Keras model</strong> with multiple <strong>LSTM layers</strong> to learn temporal patterns in the sign gestures.</li>
                    <li>Trains the model using the collected data.</li>
                    <li>Saves the trained model as <code class="text-sm font-semibold">action.h5</code>.</li>
                </ul>
            </li>
            <li>
                <strong>Real-time Translation (<code class="text-sm font-semibold">realtime_translation.py</code>):</strong>
                <ul class="list-disc pl-6 mt-2 space-y-1">
                    <li>Loads the pre-trained <code class="text-sm font-semibold">action.h5</code> model.</li>
                    <li>Continuously captures live video from your webcam.</li>
                    <li>Applies MediaPipe to extract hand keypoints from each frame.</li>
                    <li>Feeds sequences of live keypoints to the trained LSTM model for prediction.</li>
                    <li>Applies a smoothing algorithm to stabilize predictions.</li>
                    <li>Displays the recognized English word on the screen in real-time.</li>
                </ul>
            </li>
        </ol>

  <h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">📂 Project Structure</h2>
        <pre class="mb-8"><code>.
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
</code></pre>
        <h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">🛠️ Getting Started</h2>
        <h3 class="text-2xl sm:text-3xl font-semibold mb-3 text-gray-800">Prerequisites</h3>
        <ul class="list-disc pl-6 mb-8 text-gray-700 space-y-2">
            <li><strong>Python 3.9+</strong> installed.</li>
            <li><code class="text-sm font-semibold">pip</code> (Python package installer).</li>
            <li>A working <strong>webcam</strong>.</li>
        </ul>

 <h3 class="text-2xl sm:text-3xl font-semibold mb-3 text-gray-800">Installation</h3>
        <ol class="list-decimal pl-6 mb-8 text-gray-700 space-y-4">
            <li>
                <strong>Clone the repository:</strong>
                <pre><code class="language-bash">git clone https://github.com/ayushjain2729/ASL_to_Eng.git
cd ASL_to_Eng.git
</code></pre>
            </li>
            <li>
                <strong>Create a virtual environment (recommended):</strong>
                <pre><code class="language-bash">python -m venv venv
</code></pre>
            </li>
            <li>
                <strong>Activate the virtual environment:</strong>
                <ul class="list-disc pl-6 mt-2 space-y-1">
                    <li><strong>Windows:</strong>
                        <pre><code class="language-bash">.\venv\Scripts\activate
</code></pre>
                    </li>
                    <li><strong>macOS/Linux:</strong>
                        <pre><code class="language-bash">source venv/bin/activate
</code></pre>
                    </li>
                </ul>
            </li>
            <li>
                <strong>Install dependencies:</strong>
                <pre><code class="language-bash">pip install opencv-python numpy mediapipe tensorflow scikit-learn
</code></pre>
            </li>
        </ol>
<h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">🚀 Usage Guide</h2>
        <p class="text-gray-700 mb-4">Follow these steps in order to collect data, train your model, and run the real-time translator.</p>

<h3 class="text-2xl sm:text-3xl font-semibold mb-3 text-gray-800">Step 1: Collect Hand Landmark Data</h3>
        <p class="text-gray-700 mb-4">This script will guide you through collecting 30 sequences (videos) for each sign. Ensure your hands are clearly visible in the camera frame.</p>
        <ol class="list-decimal pl-6 mb-8 text-gray-700 space-y-4">
            <li>
                <strong>Important: Delete previous data (if any).</strong>
                <p class="mb-2">If you previously collected full-body data or want to start fresh, delete the <code class="text-sm font-semibold">MP_Data</code> folder before running:</p>
                <pre><code class="language-bash">rm -rf MP_Data  # For macOS/Linux
rmdir /s /q MP_Data # For Windows
</code></pre>
            </li>
            <li>
                <strong>Run the data collection script:</strong>
                <pre><code class="language-bash">python data_collection.py
</code></pre>
                <ul class="list-disc pl-6 mt-2 space-y-1">
                    <li>A webcam feed window will open.</li>
                    <li>Follow the on-screen prompts: "STARTING COLLECTION" will give you 2 seconds to prepare.</li>
                    <li>Perform the sign clearly and consistently for each of the 30 sequences per word.</li>
                    <li>Press <code class="text-sm font-semibold">q</code> or <code class="text-sm font-semibold">ESC</code> to stop collection early.</li>
                    <li>The script will close automatically once all data is collected.</li>
                </ul>
            </li>
        </ol>

 <h3 class="text-2xl sm:text-3xl font-semibold mb-3 text-gray-800">Step 2: Train the LSTM Model</h3>
        <p class="text-gray-700 mb-4">This script will train your AI model using the collected hand landmark data.</p>
        <ol class="list-decimal pl-6 mb-8 text-gray-700 space-y-4">
            <li>
                <strong>Crucial: Update Model Input Shape!</strong>
                <p class="mb-2">Since we are now using <em>only</em> hand data, the input shape of the LSTM model must be updated. Open <code class="text-sm font-semibold">model_training.py</code> and change the <code class="text-sm font-semibold">input_shape</code> in the first LSTM layer:</p>
                <pre><code class="language-python"># Find this line:
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, 1662)))
# Change it to:
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, 126)))
</code></pre>
                <p class="mt-2">(The <code class="text-sm font-semibold">126</code> comes from <code class="text-sm font-semibold">(21 left hand landmarks * 3 coordinates) + (21 right hand landmarks * 3 coordinates) = 63 + 63 = 126</code>)</p>
            </li>
            <li>
                <strong>Run the model training script:</strong>
                <pre><code class="language-bash">python model_training.py
</code></pre>
                <ul class="list-disc pl-6 mt-2 space-y-1">
                    <li>You will see output detailing data loading, model summary, and epoch-by-epoch training progress.</li>
                    <li>This process can take some time depending on your hardware.</li>
                    <li>Upon successful completion, an <code class="text-sm font-semibold">action.h5</code> file will be saved in your project directory.</li>
                </ul>
            </li>
        </ol>

<h3 class="text-2xl sm:text-3xl font-semibold mb-3 text-gray-800">Step 3: Real-time Sign Language Translation</h3>
        <p class="text-gray-700 mb-4">Now, put your trained model to the test!</p>
        <ol class="list-decimal pl-6 mb-8 text-gray-700 space-y-4">
            <li>
                <strong>Run the real-time translation script:</strong>
                <pre><code class="language-bash">python realtime_translation.py
</code></pre>
                <ul class="list-disc pl-6 mt-2 space-y-1">
                    <li>A webcam feed window will open.</li>
                    <li>Perform the signs ('hello', 'thanks', 'iloveyou') you trained the model on.</li>
                    <li>The recognized English word should appear at the top of the screen.</li>
                    <li>Press <code class="text-sm font-semibold">q</code> or <code class="text-sm font-semibold">ESC</code> to quit the application.</li>
                </ul>
            </li>
        </ol>

<h3 class="text-2xl sm:text-3xl font-semibold mb-3 text-gray-800">Troubleshooting Webcam Issues (if <code class="text-sm font-semibold">realtime_translation.py</code> exits immediately)</h3>
        <p class="text-gray-700 mb-4">If the <code class="text-sm font-semibold">realtime_translation.py</code> script closes immediately, it's almost always a webcam access issue. The script includes enhanced error messages to help you diagnose.</p>
        <ul class="list-disc pl-6 mb-8 text-gray-700 space-y-2">
            <li><strong>Close other applications:</strong> Ensure no other software (Zoom, Teams, browser tabs, other camera apps) is using your webcam.</li>
            <li><strong>Restart your computer:</strong> This often resolves lingering camera locks.</li>
            <li><strong>Check camera connection:</strong> For external webcams, ensure it's securely plugged in. Try a different USB port.</li>
            <li><strong>Privacy settings:</strong> Verify your operating system's camera privacy settings allow Python or your terminal/VS Code to access the webcam.
                <ul class="list-disc pl-6 mt-2 space-y-1">
                    <li><strong>Windows:</strong> <code class="text-sm font-semibold">Settings</code> &gt; <code class="text-sm font-semibold">Privacy &amp; security</code> &gt; <code class="text-sm font-semibold">Camera</code>.</li>
                    <li><strong>macOS:</strong> <code class="text-sm font-semibold">System Settings</code> &gt; <code class="text-sm font-semibold">Privacy &amp; Security</code> &gt; <code class="text-sm font-semibold">Camera</code>.</li>
                </ul>
            </li>
            <li><strong>Camera Index:</strong> The script tries indices 0, 1, 2. If you have a unique setup, you might need to manually adjust <code class="text-sm font-semibold">cv2.VideoCapture(i)</code> in <code class="text-sm font-semibold">realtime_translation.py</code>.</li>
        </ul>

 <h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">🔮 Future Enhancements</h2>
        <ul class="list-disc pl-6 mb-8 text-gray-700 space-y-2">
            <li><strong>Expand Vocabulary:</strong> Add more signs and corresponding data.</li>
            <li><strong>Improved UI:</strong> Develop a more user-friendly graphical interface (e.g., using PyQt, Tkinter, or a web framework).</li>
            <li><strong>Advanced Smoothing:</strong> Implement more sophisticated temporal smoothing algorithms for predictions.</li>
            <li><strong>Multi-person Detection:</strong> Extend to recognize signs from multiple individuals simultaneously.</li>
            <li><strong>Deployment:</strong> Explore deploying the model to a web service or mobile application.</li>
        </ul>

 <h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">🤝 Contributing</h2>
        <p class="text-gray-700 mb-8">Contributions are welcome! If you have suggestions, bug reports, or want to add new features, please feel free to open an issue or submit a pull request.</p>

 <h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">📜 License</h2>
        <p class="text-gray-700 mb-8">This project is licensed under the MIT License - see the <code class="text-sm font-semibold">LICENSE</code> file for details.</p>

<h2 class="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">🙏 Acknowledgements</h2>
        <ul class="list-disc pl-6 mb-8 text-gray-700 space-y-2">
            <li><strong>Google MediaPipe:</strong> For powerful and easy-to-use pose, face, and hand landmark detection.</li>
            <li><strong>TensorFlow & Keras:</strong> For the robust deep learning framework.</li>
            <li><strong>OpenCV:</strong> For camera interaction and image processing.</li>
        </ul>
    </div>
</body>
</html>
