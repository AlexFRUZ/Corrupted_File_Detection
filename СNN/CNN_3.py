import os
import pickle
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers, models, Input, Model
from keras.src.utils import to_categorical

current_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(current_dir, "..", "static", "images")
key_file = os.path.join(current_dir, "..", "key1.txt")
num_classes = 3

def load_true_labels(key_file):
    with open(key_file, 'r') as f:
        return [int(line.strip()) for line in f.readlines()]

labels = load_true_labels(key_file)
labels_categorical = to_categorical(labels, num_classes)

def load_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img) / 255.0

def denoise_image(image):
    image_uint8 = (image * 255).astype(np.uint8)
    return cv2.medianBlur(image_uint8, 5) / 255.0

def load_all_images(image_files, image_dir, target_size=(256, 256)):
    images = [denoise_image(load_image(os.path.join(image_dir, f), target_size)) for f in image_files]
    return np.array(images)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
images_data = load_all_images(image_files, image_dir)

with open("features.pkl", "rb") as f:
    valid_file_names, valid_features, cluster_labels = pickle.load(f)

cluster_features = to_categorical(cluster_labels, num_classes=3)

X_train_img, X_val_img, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
    images_data, cluster_features, labels_categorical, test_size=0.2, random_state=42
)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(labels_categorical, axis=1)),
    y=np.argmax(labels_categorical, axis=1)
)

class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

image_input = Input(shape=(256, 256, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)

cluster_input = Input(shape=(num_classes,))
combined = layers.Concatenate()([x, cluster_input])

dense_layer = layers.Dense(128, activation='relu')(combined)
output = layers.Dense(num_classes, activation='softmax')(dense_layer)

model = Model(inputs=[image_input, cluster_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    [X_train_img, X_train_feat], y_train,
    epochs=10, batch_size=32, 
    validation_data=([X_val_img, X_val_feat], y_val),
    class_weight=class_weight_dict  
)

model_dir = os.path.join(current_dir, "..", "models")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "combined_model12.h5"))

all_images = np.concatenate((X_train_img, X_val_img), axis=0)
all_features = np.concatenate((X_train_feat, X_val_feat), axis=0)
all_labels = np.concatenate((y_train, y_val), axis=0)

y_pred_probs = model.predict([all_images, all_features])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(all_labels, axis=1)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Nominal', 'Transition', 'Faulty'],
                yticklabels=['Nominal', 'Transition', 'Faulty'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_true, y_pred)

print(classification_report(y_true, y_pred, target_names=['Nominal', 'Transition', 'Faulty']))  