import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ======================
# Load Dataset
# ======================

X = np.load("X.npy")
y = np.load("y.npy")

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training:", X_train.shape)
print("Testing :", X_test.shape)

# ======================
# CNN Model
# ======================

model = Sequential([

    Conv2D(32,(3,3),activation="relu",padding="same",
           input_shape=(128,128,1)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64,(3,3),activation="relu",padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128,(3,3),activation="relu",padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(256,(3,3),activation="relu",padding="same"),
    BatchNormalization(),
    MaxPooling2D(),

    GlobalAveragePooling2D(),

    Dense(256,activation="relu"),
    Dropout(0.5),

    Dense(128,activation="relu"),
    Dropout(0.3),

    Dense(8,activation="softmax")

])

# ======================
# Compile
# ======================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ======================
# Callbacks
# ======================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    verbose=1

)

checkpoint = ModelCheckpoint(

    "best_emotion_model.keras",

    save_best_only=True,

    monitor="val_accuracy"

)

# ======================
# Train
# ======================

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test,y_test),

    epochs=100,

    batch_size=32,

    callbacks=[

        early_stop,

        reduce_lr,

        checkpoint

    ]

)

# ======================
# Evaluate
# ======================

loss, accuracy = model.evaluate(X_test,y_test)

print("\nFinal Accuracy:",accuracy)