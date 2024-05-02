# ML-Project-TRI-3

![image](https://github.com/sdjbabin/ML-Project-TRI-3/assets/137878044/015cef75-13fe-4177-be53-1779313ffa5b)

**AstroML: A study on Machine Learning tools for analyzing Astronomical Data**

This repository contains code and documentation for a comprehensive analysis of astronomical data, focusing on the classification of stars and galaxies using various machine learning and data analysis techniques. The analysis covers three main areas: stellar classification, star clustering, and galaxy morphological classification.

## Stellar Classification

### Overview
This section presents a detailed analysis of stellar classification using machine learning techniques on a dataset of stellar properties. 

### Steps
1. **Data Preprocessing**: Missing values handling and categorical variable encoding were performed.
2. **Exploratory Data Analysis (EDA)**: Histograms, density plots, box plots, and scatter plots were utilized to understand data distributions and relationships.
3. **Machine Learning Pipeline**: XGBoost classifier was employed for classification, with hyperparameter tuning and feature reduction techniques applied.
4. **Model Evaluation**: Classification accuracy, confusion matrices, and additional metrics were calculated to evaluate model performance.

## Star Clustering

### Overview
This section explores clustering techniques to analyze star properties and demarcate star clusters.

### Steps
1. **Data Preparation**: Data from popular star clusters such as NGC 188 and M67 were used.
2. **Clustering Methods**: K-means, Agglomerative Hierarchical, and DBSCAN clustering algorithms were applied.
3. **Evaluation**: Clustering results were evaluated based on separation of known star clusters and noise identification.

## Galaxy Morphological Classification

### Overview
This section focuses on classifying galaxy images into predefined morphological forms using deep learning architectures.

### Steps
1. **Data Preparation**: Galaxy10 dataset containing images of galaxies was utilized.
2. **Feature Extraction**: Histogram of Oriented Gradients (HOG) descriptors were extracted.
3. **Deep Learning Models**: MultiLayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) were employed for classification.
4. **Performance Enhancement**: Techniques such as data augmentation and dimensionality reduction (PCA) were applied.
5. **Evaluation**: Classification accuracy and performance metrics were analyzed.

## CNN on Space Images
## Overview
The Galaxy10 dataset is aimed at classifying images of galaxies into one of ten distinct shapes, including various disk orientations and smooth shapes with different characteristics. Traditional approaches in object recognition typically use feature descriptors to compress image properties into smaller vectors, which are then utilized as features for machine learning models. This study demonstrates the application of this pipeline on the Galaxy10 dataset using Histogram of Oriented Gradients (HOG) as a feature descriptor, a popular method in computer vision for object detection and classification.

### Methodology
1. **Feature Extraction with HOG**: 
   - Histogram of Oriented Gradients (HOG) is applied to extract features from galaxy images. This method computes gradients' magnitudes and orientations in localized portions of the image, providing a compact representation of the image's texture and shape.
   
2. **Neural Network Architecture**:
   - The architecture consists of a MultiLayer Perceptron (MLP) built using PyTorch's Sequential API.
   - The MLP is designed with multiple layers including Linear (fully connected) layers and non-linear activation functions (e.g., ReLU).
   - The network learns to approximate complex functions and capture non-linear relationships within the data, enabling effective classification of galaxy shapes.

3. **Training and Evaluation**:
   - The network is trained on the Galaxy10 dataset, with images represented by their HOG descriptors.
   - Training involves optimizing the network's parameters using techniques like stochastic gradient descent (SGD) or Adam optimization.
   - Model performance is evaluated on a separate validation set to assess its accuracy in classifying galaxy shapes.

### Results
- **Initial Accuracy**: 
   - The initial accuracy of the model before applying Principal Component Analysis (PCA) was 77%.

- **Accuracy Improvement with PCA**:
   - PCA was applied to further improve the accuracy, resulting in an increase from 77% to 80%.
   - 
### Usage
1. **Setup**:
   - Install necessary dependencies (PyTorch, scikit-learn, etc.).
   - Download or prepare the Galaxy10 dataset.

2. **Feature Extraction**:
   - Extract HOG descriptors from galaxy images using the provided scripts in src/.
   
3. **Model Training**:
   - Train the MLP model using PyTorch's Sequential API with the extracted HOG features.
   
4. **Evaluation**:
   - Evaluate the trained model on the validation set to measure accuracy.

5. **PCA Application**:
   - Apply PCA to further enhance accuracy if desired.
  
### Steps
1. **Data Preparation**: Data from popular star clusters such as NGC 188 and M67 were used.
2. **Clustering Methods**: K-means, Agglomerative Hierarchical, and DBSCAN clustering algorithms were applied.
3. **Evaluation**: Clustering results were evaluated based on separation of known star clusters and noise identification.

## Contributors
- Ritwika Das Gupta
- Soham Chatterjee

