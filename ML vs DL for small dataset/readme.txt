# ML vs DL: Cancer Severity Prediction

Welcome to the ML vs DL Cancer Severity Prediction project! This repository contains a Jupyter notebook demonstrating a comparison between Machine Learning (ML) and Deep Learning (DL) models on a small dataset, achieving 100% accuracy for both approaches. The dataset used in this project contains information about cancer patients with the target variable indicating the level of severity (low, medium, high).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to compare the performance of a Random Forest classifier (ML) and a Neural Network (DL) on a small dataset. Both models achieved 100% accuracy, demonstrating the potential of both approaches in handling small datasets effectively.

## Dataset

The dataset used for this project is a cancer patient dataset with various features and a target variable indicating the severity level (low, medium, high). The dataset is preprocessed to handle missing values and encode categorical variables.

### Dataset Features

- Patient age
- Tumor size
- Number of lymph nodes
- Histological type
- Hormone receptor status
- ... and more

The target variable indicates the severity level:
- Low
- Medium
- High

## Preprocessing

To ensure the dataset is ready for modeling, the following preprocessing steps were performed:

1. **Handle Missing Values:** Remove rows with missing values.
2. **Encode Categorical Variables:** Convert categorical variables into numerical format using techniques such as one-hot encoding.
3. **Feature Scaling:** Standardize the numerical features for better performance in models.

### Example of Preprocessing Code

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
data = pd.read_csv('cancer_patient_data.csv')

# Handle missing values
data.dropna(inplace=True)

# Encode categorical variables
encoder = OneHotEncoder()
categorical_features = data.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(data, columns=categorical_features)

# Feature scaling
scaler = StandardScaler()
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])
```

## Models

### Machine Learning Model

A Random Forest Classifier was used as the ML model.

### Example of Random Forest Model Code

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X = data.drop('severity', axis=1)
y = data['severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
```

### Deep Learning Model

A Neural Network was used as the DL model.

### Example of Neural Network Model Code

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Evaluate the model
nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test)
print(f"Neural Network Accuracy: {nn_accuracy * 100:.2f}%")
```

## Results

Both models achieved 100% accuracy on the test set.

- **Random Forest Accuracy:** 100%
- **Neural Network Accuracy:** 100%

These results demonstrate the effectiveness of both machine learning and deep learning approaches on small, well-preprocessed datasets.

## Conclusion

The project successfully demonstrates that both ML and DL models can achieve perfect accuracy on small, well-processed datasets. This highlights the importance of data preprocessing and the potential of both approaches in handling small datasets.

## Installation

To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/ml-vs-dl-cancer-severity-prediction.git
cd ml-vs-dl-cancer-severity-prediction
pip install -r requirements.txt
```

### Setting Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Here's how you can set it up:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

To run the project, simply execute the Jupyter notebook provided in the repository. The notebook includes steps for data preprocessing, model training, and evaluation.

```bash
jupyter notebook ML_vs_DL_Cancer_Severity_Prediction.ipynb
```

### Example Usage in Script

You can also integrate specific functions from the notebook into your Python scripts as needed.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
