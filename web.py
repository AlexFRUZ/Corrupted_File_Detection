from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from PIL import Image
from keras.src.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score

app = Flask(__name__)
app_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(app_dir)

model_paths = {
    '2': "./models/graph_cnn_model1.h5",
    '3': "./models/combined_model1.h5"
}

features_path = "features.pkl"
image_dir = "./static/images"

kmeans_paths = {
    '2': "./models/kmeans_model2.pkl",
    '3': "./models/kmeans_model3.pkl"
}

directory = "zeroShot_vkiit"
file_list = [f'T{str(i).zfill(4)}.txt' for i in range(1, 1159)]
key_file_path = "key.txt"

models = {key: load_model(path) for key, path in model_paths.items()}

try:
    with open(features_path, 'rb') as f:
        valid_file_names, valid_features, cluster_labels = pickle.load(f)
except FileNotFoundError:
    print(f"File {features_path} not found. Check the file path.")
    valid_file_names, valid_features, cluster_labels = [], [], []

def load_true_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [float(line.strip()) for line in file.readlines()]
    return labels

def load_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = np.array([float(line.strip()) for line in f])
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def get_cluster_features(file_name):
    try:
        if len(valid_file_names) == 0:
            raise ValueError("Cluster features file not loaded or empty.")
        cluster_idx = np.where(valid_file_names == file_name)[0]
        if cluster_idx.size > 0:
            return valid_features[cluster_idx[0]]
        else:
            raise ValueError("File not found in the list.")
    except ValueError:
        return np.zeros((3,))

def load_and_preprocess_image_3(file, target_size=(256, 256)):
    img = Image.open(file).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def load_and_preprocess_image_2(file, target_size=(128, 128)):
    img = Image.open(file).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image_with_cnn(file, model, optimal_classes):
    if optimal_classes == '3':
        img = load_and_preprocess_image_3(file)
    else:
        img = load_and_preprocess_image_2(file)
    predictions = model.predict(img)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    file_path = os.path.join(image_dir, file.filename)
    file.save(file_path)

    optimal_classes = request.form.get('classes')
    if optimal_classes not in models:
        return 'Invalid number of classes', 400

    model = models[optimal_classes]

    if optimal_classes == '3':
        cluster_features = get_cluster_features(file.filename)
        cluster_features = np.expand_dims(cluster_features, axis=0)
        
        img = load_and_preprocess_image_3(file_path)
        predictions = model.predict([img, cluster_features])
        predicted_class = np.argmax(predictions, axis=1)[0]

        if predicted_class == 0:
            predicted_class_label = 'Nominal data'
            message = "This is normal data."
        elif predicted_class == 1:
            predicted_class_label = 'Transition data'
            message = "You should pay more attention to this data because it is on the verge of breaking."
        else:
            predicted_class_label = 'Fault data'
            message = "This is corrupted data."
        
    else:
        img = load_and_preprocess_image_2(file_path)
        predictions = model.predict(img)

        prediction_label = 'Nominal data' if predictions[0] > 0.5 else 'Fault data'
        if prediction_label == 'Nominal data':
            predicted_class_label = 'Nominal data'
            message = "This is normal data."
        else:
            predicted_class_label = 'Fault data'
            message = "This is corrupted data."
    if optimal_classes == '2':
        predicted_probabilities = predictions[0].tolist()
        if prediction_label == 'Nominal data':
            predicted_probabilities = max(predicted_probabilities)
        else:
            predicted_probabilities = 1 - max(predicted_probabilities)
    else:
        predicted_probabilities = predictions[0].tolist()
        predicted_probabilities = max(predicted_probabilities)
    return render_template('results.html', 
                       image_path=file.filename, 
                       predicted_class=predicted_class_label, 
                       predicted_probabilities=predicted_probabilities, 
                       message=message,
                       optimal_classes=optimal_classes)

@app.route('/predict', methods=['POST'])
def predict():
    cluster_type = request.form.get('clusters')
    if not cluster_type:
        return jsonify({'error': 'No cluster type provided.'}), 400

    optimal_clusters = int(cluster_type)

    kmeans_model_path = kmeans_paths.get(str(optimal_clusters))

    if kmeans_model_path is None:
        return jsonify({'error': 'Invalid number of clusters.'}), 400

    with open(kmeans_model_path, 'rb') as f:
        kmeans = pickle.load(f)

    features = []
    file_names = []
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        data = load_file(file_path)
        if data is not None and len(data) > 0:
            mean = np.mean(data)
            median = np.median(data)
            std = np.std(data)
            data_min = np.min(data)
            data_max = np.max(data)
            features.append([mean, median, std, data_min, data_max])
            file_names.append(file_name)
        else:
            features.append([np.nan, np.nan, np.nan, np.nan, np.nan])
            file_names.append(file_name)

    features = np.array(features)
    valid_indices = ~np.isnan(features).any(axis=1)
    valid_features = features[valid_indices]
    valid_file_names = np.array(file_names)[valid_indices]

    if valid_features.shape[0] == 0:
        return jsonify({'error': 'No valid samples found for clustering.'}), 400

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(valid_features)

    labels = kmeans.predict(scaled_features)

    true_labels = load_true_labels(key_file_path)
    silhouette_avg = silhouette_score(scaled_features, labels)
    davies_bouldin = davies_bouldin_score(scaled_features, labels)
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)

    metrics = {
        'silhouette_avg': silhouette_avg,
        'davies_bouldin': davies_bouldin,
        'ari': ari,
        'nmi': nmi
    }

    return render_template('index.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
