from predict import predict_emotion

audio_file = r"..\dataset\Actor_01\03-01-03-02-02-01-01.wav"

emotion, confidence = predict_emotion(audio_file)

print("Predicted Emotion:", emotion)
print(f"Confidence: {confidence:.2f}%")