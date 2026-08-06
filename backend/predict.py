import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

model = load_model("best_emotion_model.keras")
encoder = joblib.load("label_encoder.pkl")

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    if mel_db.shape[1] < 128:
        pad = 128 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")

    mel_db = mel_db[:, :128]

    mel_db = mel_db.reshape(1, 128, 128, 1)

    return mel_db


def predict_emotion(file_path):
    features = preprocess_audio(file_path)

    prediction = model.predict(features, verbose=0)

    predicted_index = np.argmax(prediction)

    emotion = encoder.inverse_transform([predicted_index])[0]

    confidence = float(np.max(prediction) * 100)

    return emotion, confidence