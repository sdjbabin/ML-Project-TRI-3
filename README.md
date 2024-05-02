# ML-Project-TRI-3

![image](https://github.com/sdjbabin/ML-Project-TRI-3/assets/137878044/015cef75-13fe-4177-be53-1779313ffa5b)

**GitHub README File: Comprehensive Analysis of Astronomical Data**

---

# Comprehensive Analysis of Astronomical Data

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

## Repository Structure
- **/stellar_classification**: Code and documentation for stellar classification analysis.
- **/star_clustering**: Code and documentation for star clustering analysis.
- **/galaxy_classification**: Code and documentation for galaxy morphological classification analysis.
- **/data**: Dataset files used in the analysis.
- **/images**: Images and visualizations generated during the analysis.
- **README.md**: Main README file providing an overview of the repository.

## Requirements
- Python 3.x
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, xgboost, torch, torchvision

## Usage
1. Clone the repository.
2. Navigate to the desired analysis directory.
3. Run the Jupyter notebooks or Python scripts to reproduce the analysis.

## Contributors
- John Doe (@johndoe)
- Jane Smith (@janesmith)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize and expand upon this README file to suit your project's specific details and requirements.
