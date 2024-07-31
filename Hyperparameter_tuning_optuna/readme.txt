# Deep Learning Hyperparameter Tuning with Optuna

Welcome to the Deep Learning Hyperparameter Tuning project! This repository contains a Jupyter notebook demonstrating the use of Optuna for hyperparameter tuning in a deep learning model. As a machine learning developer and data scientist, this project will showcase your ability to optimize model performance through systematic tuning of hyperparameters.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates hyperparameter tuning using Optuna on a deep learning model. The notebook leverages TensorFlow and TensorFlow Hub for model building and training, focusing on optimizing the model's performance through careful adjustment of hyperparameters.

## Installation

To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/deep-learning-hyperparameter-tuning.git
cd deep-learning-hyperparameter-tuning
pip install -r requirements.txt
```

## Usage

To run the project, simply execute the Jupyter notebook provided in the repository. The notebook includes steps for data loading, model building, hyperparameter tuning with Optuna, and evaluation.

```bash
jupyter notebook Hyperparameter_tuning_optuna.ipynb
```

### Example Usage in Script

You can also integrate specific functions from the notebook into your Python scripts as needed.

## Dataset

The dataset used in this project is the Flower Photos dataset, which can be downloaded directly using TensorFlow utilities:

```python
data_dir = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
```

### Building the Dataset

The dataset is built using the `tf.keras.preprocessing.image_dataset_from_directory` function:

```python
def build_dataset(subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.20,
        subset=subset,
        label_mode="categorical",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=32)
```

## Model Architecture

The model uses a pre-trained EfficientNetV2 model from TensorFlow Hub as the backbone for feature extraction. The architecture includes:

- EfficientNetV2 backbone
- Custom dense layers for classification

### Example Model Definition

```python
model_name = "efficientnetv2-xl-21k-ft1k"
model_handle_map = {
    "efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"
}
model_handle = model_handle_map[model_name]
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 32

model = tf.keras.Sequential([
    hub.KerasLayer(model_handle, input_shape=IMAGE_SIZE + (3,), trainable=True),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
```

## Hyperparameter Tuning

The hyperparameter tuning is conducted using Optuna, an automatic hyperparameter optimization framework:

### Example Hyperparameter Tuning with Optuna

```python
import optuna

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    
    model = tf.keras.Sequential([
        hub.KerasLayer(model_handle, input_shape=IMAGE_SIZE + (3,), trainable=True),
        tf.keras.layers.Dropout(rate=dropout_rate),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
    val_accuracy = history.history['val_accuracy'][-1]
    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## Results

The results of the hyperparameter tuning are captured and displayed, highlighting the best hyperparameters and the model's performance.

### Example of Results Output

```python
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation accuracy: {study.best_value:.4f}")
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
