import librosa
import numpy as np

IMG_HEIGHT = 128
IMG_WIDTH = 128

def extract_mel_spectrogram(file_path):

    audio, sr = librosa.load(file_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    # Pad if needed
    if mel_db.shape[1] < IMG_WIDTH:
        pad = IMG_WIDTH - mel_db.shape[1]
        mel_db = np.pad(
            mel_db,
            ((0,0),(0,pad)),
            mode="constant"
        )

    mel_db = mel_db[:, :IMG_WIDTH]

    return mel_db