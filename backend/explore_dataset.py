import matplotlib
matplotlib.use("Agg")

import librosa
import librosa.display
import matplotlib.pyplot as plt

file_path = "../dataset/Actor_01/03-01-01-01-01-01-01.wav"

audio, sample_rate = librosa.load(file_path, sr=None)

plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio, sr=sample_rate)

plt.title("Speech Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.savefig("waveform.png")

print("Waveform saved successfully as waveform.png")