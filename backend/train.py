import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Dataset path
DATASET_PATH = "../dataset"

emotion_dict = {
    "01":"Neutral",
    "02":"Calm",
    "03":"Happy",
    "04":"Sad",
    "05":"Angry",
    "06":"Fearful",
    "07":"Disgust",
    "08":"Surprised"
}

X = []
y = []

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)


for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            emotion = emotion_dict[file.split("-")[2]]

            features = extract_features(
                os.path.join(actor_path,file)
            )

            X.append(features)

            y.append(emotion)

X=np.array(X)

encoder=LabelEncoder()

y=encoder.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:",len(X_train))
print("Testing Samples:",len(X_test))

model = Sequential()

model.add(Dense(256,activation="relu",input_shape=(40,)))

model.add(Dropout(0.3))

model.add(Dense(128,activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(64,activation="relu"))

model.add(Dense(8,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history=model.fit(

    X_train,
    y_train,

    validation_data=(X_test,y_test),

    epochs=50,

    batch_size=32
)

loss,accuracy=model.evaluate(X_test,y_test)

print()

print("Test Accuracy:",accuracy)