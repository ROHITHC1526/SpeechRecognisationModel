import os
import librosa
import numpy as np

# Dataset path
DATASET_PATH = "../dataset"

# Emotion mapping
emotion_dict = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

X = []
y = []

# MFCC extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


# Read every actor folder
for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    # Read every audio file
    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            file_path = os.path.join(actor_path, file)

            # Filename format:
            # 03-01-05-01-02-02-12.wav

            emotion_code = file.split("-")[2]

            emotion = emotion_dict[emotion_code]

            features = extract_features(file_path)

            X.append(features)

            y.append(emotion)

print("Total Samples:", len(X))

print("Feature Shape:", X[0].shape)

print("First Emotion:", y[0])