import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import extract_mel_spectrogram

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

print("Loading Dataset...")

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            mel = extract_mel_spectrogram(
                os.path.join(actor_path,file)
            )

            X.append(mel)

            emotion=file.split("-")[2]

            y.append(emotion_dict[emotion])

X=np.array(X)

X=X[...,np.newaxis]

encoder=LabelEncoder()

y=encoder.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training:",len(X_train))
print("Testing:",len(X_test))

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,1)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(
    Conv2D(
        128,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(8,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_test,y_test),

    epochs=30,

    batch_size=32,

    callbacks=[early_stop]

)

loss,accuracy=model.evaluate(X_test,y_test)

print("\nAccuracy:",accuracy)

model.save("emotion_cnn.keras")

print("CNN Model Saved Successfully")