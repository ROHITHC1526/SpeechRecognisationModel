# Speech Emotion Recognition System

## 📌 Overview
This project is a complete web application designed to predict human emotion from audio speech files. It consists of a **FastAPI backend** for handling machine learning inference and a **React (Vite) frontend** for an intuitive user interface. The core machine learning model uses a Deep Convolutional Neural Network (CNN) trained on Mel Spectrogram representations of audio data.

---

## ✨ Features
*   **Web Interface**: Clean, modern UI built with React, Vite, and Ant Design. Allows users to upload `.wav` files and see the predicted emotion along with a confidence score.
*   **Fast API Backend**: High-performance backend utilizing FastAPI to process audio uploads and run model predictions via the `/predict` endpoint.
*   **Deep Learning Model**: A robust 2D CNN architecture utilizing `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `GlobalAveragePooling2D`, and `Dropout` layers.
*   **Audio Processing**: Utilizes `librosa` to extract 128x128 Mel Spectrograms from audio waveforms, converting them to decibel scale and normalizing them.
*   **Data Augmentation**: To prevent overfitting and build a robust model, the dataset is augmented using:
    *   Noise Injection
    *   Pitch Shifting
    *   Time Stretching
*   **8 Emotion Classes**: Recognizes Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised (follows the RAVDESS dataset format).

---

## 📂 Project Structure
*   `backend/`: Contains the Python FastAPI server (`app.py`), model training scripts (`train_cnn_v2.py`), audio processing and augmentation logic (`prepare_dataset_v2.py`), and the saved keras model (`best_emotion_model.keras`).
*   `frontend/`: Contains the React/Vite web application source code.
*   `dataset/`: The directory where raw audio `.wav` files (organized by actor folders) should be placed for training.

---

## 📊 Model & Accuracy
The model relies on a deep CNN processing Mel Spectrogram images. 
During training, the model uses:
*   **Adam Optimizer** with an initial learning rate of `0.001`.
*   **Sparse Categorical Crossentropy** loss function.
*   **Early Stopping** (patience of 10) to restore the best weights and prevent overfitting.
*   **ReduceLROnPlateau** to halve the learning rate if the validation loss stagnates.

*Note: The exact final accuracy depends on your specific dataset volume. However, this architecture combined with extensive data augmentation represents a state-of-the-art approach for speech emotion recognition, designed to yield highly competitive accuracy (typically 75-85%+ on datasets like RAVDESS).*

---

## 🚀 Installation & Setup

### 1. Prerequisites
*   Python 3.8+
*   Node.js (v18+ recommended)
*   NPM or Yarn

### 2. Backend Setup
Navigate to the backend directory, install the required packages, and start the server:

```bash
cd backend

# (Optional but recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app:app --reload
```
*The backend API will run at `http://localhost:8000`.*

### 3. Frontend Setup
Navigate to the frontend directory, install dependencies, and run the development server:

```bash
cd frontend

# Install Node modules
npm install

# Start the Vite development server
npm run dev
```
*The frontend application will typically run at `http://localhost:5173`.*

---

## 🧠 How to Retrain the Model (Optional)
If you wish to train the model from scratch or use your own dataset:

1.  **Prepare the Data**: Place your `.wav` files inside the `dataset/` folder. Ensure the files follow the naming convention where the emotion identifier is the third part of the filename separated by dashes (e.g., `03-01-05-01-01-01-01.wav` where `05` is Angry).
2.  **Extract Features**:
    ```bash
    cd backend
    python prepare_dataset_v2.py
    ```
    *This will extract Mel Spectrograms, apply data augmentation, and save the data as `X.npy` and `y.npy`.*
3.  **Train the CNN**:
    ```bash
    python train_cnn_v2.py
    ```
    *The training process will begin, utilizing early stopping. The best model will be saved as `best_emotion_model.keras`.*
