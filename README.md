
# LUNG CANCER PREDICTION USING CNN AND TRANSFER LEARNING

## Project Overview

Lung cancer remains a leading cause of cancer-related deaths worldwide. This project develops a deep learning solution for automated lung cancer prediction and classification using Computed Tomography (CT) scan images.

The system utilizes custom Convolutional Neural Networks (CNNs) and advanced Transfer Learning techniques (including InceptionV3 and ResNet-50) to classify images into four distinct categories:

  * **Normal**
  * **Adenocarcinoma** (`adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`)
  * **Large Cell Carcinoma** (`large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa`)
  * **Squamous Cell Carcinoma** (`squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa`)

This automated solution provides near-instantaneous predictions, significantly improving diagnostic speed and consistency compared to traditional, subjective interpretation by radiologists

-----

## Dataset

This project uses the **original CT scan images** collected from Nagpur **Indira Gandhi Government Medical College & Hospital (IGGMCH)** (Nagpur, India). These images form the core dataset used for analysis, model training, and validation.

All CT scan data were obtained and utilized strictly for academic and research purposes, following ethical standards for data privacy and confidentiality.
Due to privacy restrictions, the original CT images are not uploaded or shared in this repository.

For reference:

The dataset consists of CT scan images categorized according to specific medical conditions.

Data preprocessing included resizing, normalization, and augmentation steps before model training.


## Key Results and Technical Highlights

The Transfer Learning approach achieved robust performance on the validation set.

  * **Final Accuracy Goal:** The model was trained to achieve an accuracy of **87%** (or higher) on the test data.
  * **Model Efficiency:** Used **InceptionV3** with its base layers frozen for efficient training and robust feature extraction.
  * **Hyperparameter Optimization:** Implemented **Early Stopping** and **Learning Rate Reduction** callbacks to prevent overfitting and ensure optimal convergence.

| Parameter | Value/Technique | Detail |
| :--- | :--- | :--- |
| **Input Image Size** | **350x350** pixels | Increased resolution for finer feature capture. |
| **Batch Size** | **8** | Used due to memory constraints and optimization. |
| **Optimizer** | **Adam Optimizer** | Default Learning Rate (`0.001`) used for compilation. |
| **Key Layers** | **GlobalAveragePooling2D** | Used to summarize feature maps before final prediction. |

-----

## Methodology: Model Architecture

The deep learning pipeline is built on the TensorFlow/Keras framework and utilizes a frozen Transfer Learning base.

### 1\. Model Definition (InceptionV3 Transfer Learning)

The classification architecture is a `Sequential` model comprising the pre-trained feature extractor and a custom classification head.

  * **Base Model:** `InceptionV3` (Pre-trained on ImageNet, `include_top=False`)
  * **Pooling Layer:** `GlobalAveragePooling2D` (Used to flatten features)
  * **Classification Head:** `Dense` layer with `softmax` activation for the 4 final classes.

### 2\. Training Strategy & Callbacks

The training process employed multiple callbacks for regularization and optimization:

  * `ReduceLROnPlateau`: Halves the learning rate if the loss does not improve after 5 epochs, preventing stalls.
  * `EarlyStopping`: Stops training if the loss does not improve after 6 epochs, preventing overfitting.
  * `ModelCheckpoint`: Saves the model weights whenever a new best loss value is achieved.

-----

## Code Implementation

The following code snippets reflect the exact implementation used for data handling and model construction.

### 1\. Data Generators (`data_prep.py`)

This section sets up the data generators, applies standard rescaling (`1/255`), and defines the image and batch sizes.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration (From Jupyter Notebook) ---
IMAGE_SIZE = (350, 350) 
BATCH_SIZE = 8
TRAIN_FOLDER = 'Data/train' 
TEST_FOLDER = 'Data/test' # Used for validation

# Define data generators: rescale applied to both
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training Generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_FOLDER,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode='categorical'
)

# Validation Generator (using the test folder as per the notebook logic)
validation_generator = test_datagen.flow_from_directory(
    TEST_FOLDER,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode='categorical'
)
```

### 2\. Model Definition and Compilation (`model_build.py`)

This code defines the InceptionV3 Transfer Learning model with its Global Average Pooling head.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Model configuration
OUTPUT_SIZE = 4 

# Load the pre-trained InceptionV3 model
pretrained_model = InceptionV3(
    weights='imagenet', 
    include_top=False, 
    input_shape=(350, 350, 3) # Note: 350x350 size used
)

# Freeze the layers
pretrained_model.trainable = False

# Build the Sequential model
model = Sequential()
model.add(pretrained_model)
model.add(GlobalAveragePooling2D()) # Custom pooling layer
model.add(Dense(OUTPUT_SIZE, activation='softmax')) # Final classification layer

# Compile the model with Adam optimizer
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()
```

### 3\. Training and Evaluation (`train_run.py`)

This script includes the optimization callbacks and executes the training loop for 50 epochs.

```python
# Define Callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.5, min_lr=0.000001)
early_stops = EarlyStopping(monitor='loss', min_delta=0, patience=6, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='best_model.weights.h5', verbose=1, save_best_only=True, save_weights_only=True)

# Start training
history = model.fit(
    train_generator,
    steps_per_epoch=25,
    epochs=50,
    callbacks=[learning_rate_reduction, early_stops, checkpointer],
    validation_data=validation_generator,
    validation_steps=20
)

# Save the final model
model.save('lung_cancer_final_model.h5')
```


Review Results: The final metrics will be displayed in the notebook output.
## END OF PROJECT DOCUMENTATION 🏁

This concludes the detailed documentation for the Bachelor of Technology thesis, "LUNG CANCER PREDICTION USING CNN AND TRANSFER LEARNING."

## IMPORTANT: The dataset is NOT included in this GitHub repository.

Due to file size and data privacy concerns, the necessary CT scan image dataset is NOT included in this repository. To fully run the Lung Cancer.ipynb notebook and reproduce the results, the reviewer must first acquire the data and structure it on their Google Drive according to the paths specified in the notebook's  DATA ACQUISITION section.
