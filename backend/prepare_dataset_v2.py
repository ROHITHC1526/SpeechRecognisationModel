import os
import numpy as np
import librosa

from preprocess import extract_mel_spectrogram
from augmentation import add_noise, pitch_shift, time_stretch

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

IMG_WIDTH = 128

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        emotion = emotion_dict[file.split("-")[2]]

        # Load original audio
        audio, sr = librosa.load(file_path, sr=22050)

        audios = [
            audio,
            add_noise(audio),
            pitch_shift(audio, sr),
            time_stretch(audio)
        ]

        for sample in audios:

            mel = librosa.feature.melspectrogram(
                y=sample,
                sr=sr,
                n_mels=128
            )

            mel = librosa.power_to_db(mel, ref=np.max)

            mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-8)

            if mel.shape[1] < IMG_WIDTH:
                pad = IMG_WIDTH - mel.shape[1]
                mel = np.pad(
                    mel,
                    ((0,0),(0,pad)),
                    mode="constant"
                )

            mel = mel[:, :IMG_WIDTH]

            X.append(mel)
            y.append(emotion)

X = np.array(X)
X = X[..., np.newaxis]
y = np.array(y)

print("Dataset Shape :", X.shape)
print("Labels Shape  :", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset Saved Successfully!")