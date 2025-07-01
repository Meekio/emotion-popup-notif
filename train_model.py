import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the FER2013 dataset
df = pd.read_csv(r"E:\emotion-popup\fer2013.csv")

X = []
y = []

for i in range(len(df)):
    pixels = np.array(df['pixels'][i].split(), dtype='float32').reshape(48, 48)
    X.append(pixels)
    y.append(df['emotion'][i])

X = np.array(X) / 255.0
X = X.reshape(-1, 48, 48, 1)
y = to_categorical(y, num_classes=7)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax')
])

# Compile & train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=64)

# Save model
model.save("model.h5")
print("✅ Model saved as model.h5")

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred_classes))
