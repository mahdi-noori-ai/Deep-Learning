
# LSTM-Based Human Activity Recognition

Welcome to the LSTM-Based Human Activity Recognition project! This repository contains a Jupyter notebook that demonstrates how to classify human activities using Long Short-Term Memory (LSTM) networks on the UCI HAR Dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Implementation](#model-implementation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to develop a deep learning model using LSTM networks to classify different human activities (e.g., walking, sitting, standing) based on data collected from smartphones' accelerometers and gyroscopes. The UCI HAR dataset is utilized for this purpose.

## Dataset

The UCI Human Activity Recognition (HAR) dataset contains data from 30 participants who performed six activities (walking, walking upstairs, walking downstairs, sitting, standing, and lying) while wearing a smartphone on their waist. The data includes sensor signals recorded by accelerometers and gyroscopes.

### Dataset Features

- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
- Triaxial Angular velocity from the gyroscope.
- 561 features, resulting from the sensor signals preprocessing.
- The target variable indicates the activity type:
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Lying

## Preprocessing

To prepare the dataset for training the LSTM model, the following preprocessing steps were applied:

1. **Loading the Data:** The dataset is downloaded and extracted automatically.
2. **Reshaping Data:** The data is reshaped into a format suitable for LSTM input, where each sample corresponds to a time series of sensor readings.
3. **Normalization:** Sensor data is normalized to improve the model's performance.

### Example of Preprocessing Code

```python
import pandas as pd
import numpy as np

# Load the dataset and preprocess it
def load_data():
    # Code to load and preprocess the dataset
    pass

X_train, X_test, y_train, y_test = load_data()

# Normalize the data
X_train = X_train / np.linalg.norm(X_train)
X_test = X_test / np.linalg.norm(X_test)
```

## Model Implementation

### LSTM Model

An LSTM network is implemented using TensorFlow/Keras to model the time-series data for activity recognition. The model architecture consists of LSTM layers followed by dense layers to predict the activity classes.

### Example of LSTM Model Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Build the LSTM model
model = Sequential([
    LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.5),
    LSTM(100),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

## Results

The LSTM model achieves high accuracy in classifying the activities. The results are visualized using a confusion matrix to understand the performance across all activity categories.

- **Model Accuracy:** The model achieved a high accuracy, reflecting its effectiveness in recognizing different activities based on sensor data.

### Example of Results Visualization Code

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on test data
y_pred = model.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

## Conclusion

This project successfully demonstrates the use of LSTM networks for human activity recognition. The model's ability to achieve high accuracy highlights the potential of deep learning in time-series data classification tasks.

## Installation

To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/lstm-human-activity-recognition.git
cd lstm-human-activity-recognition
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
jupyter notebook lstm_human_activity_recognition.ipynb
```

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

--
