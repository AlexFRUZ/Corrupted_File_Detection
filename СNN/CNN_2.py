import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import layers, models

current_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(current_dir, "..", "static", "images")
key_file = os.path.join(current_dir, "..", "key.txt")


def load_true_labels(key_file):
    with open(key_file, 'r') as f:
        lines = f.readlines()
    true_labels = [1 if float(line.strip()) == 0.0 else 0 for line in lines]
    return true_labels

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
labels = load_true_labels(key_file)

train_images, test_images, train_labels, test_labels = train_test_split(image_files, labels, test_size=0.2, random_state=42)

def load_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return img

batch_size = 32
image_size = (128, 128)

def data_generator(image_files, labels, batch_size, image_dir, target_size=(128, 128)):
    while True:
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            images = [load_image(os.path.join(image_dir, file), target_size) for file in batch_files]
            yield np.array(images), np.array(batch_labels)

train_generator = data_generator(train_images, train_labels, batch_size, image_dir, image_size)
test_generator = data_generator(test_images, test_labels, batch_size, image_dir, image_size)

steps_per_epoch = len(train_images) // batch_size
validation_steps = len(test_images) // batch_size

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=5,
    validation_data=test_generator,
    validation_steps=validation_steps
)

test_loss, test_acc = model.evaluate(test_generator, steps=validation_steps)
print(f"Test accuracy: {test_acc:.4f}")

model_dir = os.path.join(current_dir, "..", "models")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "graph_cnn_model.h5"))