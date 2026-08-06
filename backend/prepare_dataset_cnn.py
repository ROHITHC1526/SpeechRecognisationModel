import os
import numpy as np

from preprocess import extract_mel_spectrogram

DATASET_PATH = "../dataset"

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

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            file_path = os.path.join(actor_path, file)

            # Extract Mel Spectrogram
            mel = extract_mel_spectrogram(file_path)

            X.append(mel)

            # Get emotion label
            emotion = emotion_dict[file.split("-")[2]]

            y.append(emotion)

X = np.array(X)

# CNN expects (samples, height, width, channels)
X = X[..., np.newaxis]

print("Dataset Shape:", X.shape)
print("Total Samples:", len(y))
print("First Emotion:", y[0])