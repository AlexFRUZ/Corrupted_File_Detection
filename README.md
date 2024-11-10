# THE METHOD OF PREVENTING FAILURES OF ROTATING MACHINES BASED ON VIBRATION ANALYSIS BY USING MACHINE LEARNING TECHNIQUES

## Description
Repository includesThis project was developed to detect corrupted files using various approaches, including machine learning algorithms (K-means clustering) and deep learning (convolutional neural networks). To enhance usability, a web application was created on the Flask framework, providing a convenient interface for file integrity analysis

## Authors

Zalutska O.O. – Assistant of the Department of Computer Sciences, Khmelnytskyi National University, Khmelnytskyi, Ukraine.

Hladun O. V. – student of the Department of Computer Sciences, Khmelnytskyi National University, Khmelnytskyi, Ukraine.

### Folder and File Descriptions

#### 1. `models/`

Contains pre-trained (saved) models:
- **CNN** models for 2 and 3 classes.
- **K-means** models for 2 and 3 clusters.

Classes:
- For 2 classes: `Nominal data`, `Fault data`.
- For 3 classes: `Nominal data`, `Transitional data`, `Fault data`.

#### 2. `K-means/`

- **`k-means.py`**: Script implementing K-means clustering for 2 and 3 clusters, used for data clustering.

#### 3. `CNN/`

Contains code for CNN models:

- **`CNN_2.py`**: CNN model for 2-class classification (`Nominal data`, `Fault data`).
- **`CNN_3.py`**: CNN model for 3-class classification (`Nominal data`, `Transitional data`, `Fault data`).

#### 4. `zeroShot_vkiit/`

- **`initial dataset`**: A text file containing the initial dataset used for model training and classification.

#### 5. `static/`

Contains static resources:
- **Graphs**: Used for visualizing results and data on the website.
- **Initial dataset image**: Initial dataset visualized as a .png image.

#### 6. `templates/`

Contains HTML templates for the website, where classification and clustering results, as well as other interface pages, are displayed.

#### 7. `feature.pkl`

File containing features used in clustering and classification for 3 classes (`Nominal data`, `Transitional data`, `Fault data`).

#### 8. `key.txt`

File with labels for 2-class classification (`Nominal data`, `Fault data`).

#### 9. `key1.txt`

File with labels for 3-class classification (`Nominal data`, `Transitional data`, `Fault data`).

#### 10. `requirements.txt`

A list of required libraries for the project. To install dependencies, run:

```bash
pip install -r requirements.txt
