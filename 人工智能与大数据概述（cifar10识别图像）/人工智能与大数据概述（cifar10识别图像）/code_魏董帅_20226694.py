import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from time import time
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constant definitions
NUM_CLASSES = 10  # CIFAR-10 has 10 classes
BATCH_SIZE = 128   # Increase batch size
EPOCHS = 150      # Increase number of training epochs
LEARNING_RATE = 0.0005  # Decrease learning rate
PATIENCE = 20    # Increase patience for early stopping
MODEL_PATH = 'cifar10_cnn_model.h5'  # Model save path
CSV_PATH = 'sample_submission.csv'   # Prediction results CSV path
DATA_DIR = r'C:\Users\31050\Desktop\人工智能与大数据概述\作业\数据\neu-artificial'

# CIFAR-10 class names
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Data loading and preprocessing function
def load_and_preprocess_data():
    """
    Load and preprocess the dataset.
    Returns:
        (x_train, y_train): Training data and labels
        (x_val, y_val): Validation data and labels
        x_test: Test data
    """
    # Load training data
    train_labels_df = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
    x_train = []
    y_train = []
    for index, row in train_labels_df.iterrows():
        filename = row['filename']
        label = row['label']
        img_path = os.path.join(DATA_DIR, 'train', filename)
        img = load_img(img_path, target_size=(32, 32))
        img = img_to_array(img)
        x_train.append(img)
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)

    # Load validation data
    val_labels_df = pd.read_csv(os.path.join(DATA_DIR, 'val_labels.csv'))
    x_val = []
    y_val = []
    for index, row in val_labels_df.iterrows():
        filename = row['filename']
        label = row['label']
        img_path = os.path.join(DATA_DIR, 'val', filename)
        img = load_img(img_path, target_size=(32, 32))
        img = img_to_array(img)
        x_val.append(img)
        y_val.append(label)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

    # Load test data
    test_files = os.listdir(os.path.join(DATA_DIR, 'test'))
    x_test = []
    for filename in test_files:
        img_path = os.path.join(DATA_DIR, 'test', filename)
        img = load_img(img_path, target_size=(32, 32))
        img = img_to_array(img)
        x_test.append(img)
    x_test = np.array(x_test)

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Print dataset shape information
    print("\nDataset shapes:")
    print(f"Training set: {x_train.shape}, {y_train.shape}")
    print(f"Validation set: {x_val.shape}, {y_val.shape}")
    print(f"Test set: {x_test.shape}")

    return (x_train, y_train), (x_val, y_val), x_test


def plot_sample_images(images, labels, title="Sample image examples", num_samples=25):
    """
    Visualize sample images from the dataset.

    Args:
        images: Image data
        labels: Image labels
        title: Chart title
        num_samples: Number of samples to display
    """
    plt.figure(figsize=(10, 10))
    # Calculate grid size (try to be as close to a square as possible)
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    for i in range(num_samples):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i].astype(np.uint8))
        plt.title(CLASS_NAMES[np.argmax(labels[i])], fontsize=8)
        plt.axis('off')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    plt.show()


# Build the CNN model
def build_cnn_model():
    """
    Build a convolutional neural network model.

    Returns:
        model: A built Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      input_shape=(32, 32, 3), name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.2, name='dropout1'),

        # Second convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4'),
        layers.BatchNormalization(name='bn4'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.3, name='dropout2'),

        # Third convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5'),
        layers.BatchNormalization(name='bn5'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6'),
        layers.BatchNormalization(name='bn6'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.4, name='dropout3'),

        # Fully connected layers
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='fc1'),
        layers.BatchNormalization(name='bn7'),
        layers.Dropout(0.5, name='dropout4'),
        layers.Dense(NUM_CLASSES, activation='softmax', name='output')
    ])

    return model


# Compile the model
def compile_model(model):
    """
    Compile the model, configure the optimizer, loss function, and evaluation metrics.

    Args:
        model: The Keras model to be compiled
    Returns:
        The compiled model
    """
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model


# Create training callbacks
def create_callbacks():
    """
    Create callbacks to be used during training.

    Returns:
        A list of callbacks
    """
    callbacks_list = [
        # Early stopping callback: Stop training when validation accuracy stops improving
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint: Save the best model
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Learning rate adjustment: Reduce the learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        # TensorBoard callback: Record training logs
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            profile_batch=0  # Disable performance profiling to avoid slowing down training
        )
    ]
    return callbacks_list


# Plot the training history
def plot_history(history):
    """
    Visualize the accuracy and loss changes during training.

    Args:
        history: The training history object
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(18, 6))

    # Accuracy curve
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Model accuracy', fontsize=12)
    plt.ylabel('Accuracy')
    plt.xlabel('Training epochs')
    plt.legend()

    # Loss curve
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss', fontsize=12)
    plt.ylabel('Loss')
    plt.xlabel('Training epochs')
    plt.legend()

    # Learning rate change curve
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning rate')
        plt.title('Learning rate change', fontsize=12)
        plt.ylabel('Learning rate')
        plt.xlabel('Training epochs')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Learning rate not recorded', ha='center', va='center')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot the confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion matrix', fontsize=12)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


# Evaluate the model
def evaluate_model(model, x_val, y_val):
    """
    Evaluate the model performance and generate a report.

    Args:
        model: The trained model
        x_val: Validation data
        y_val: Validation labels
    """
    print("\nEvaluating model performance...")
    start_time = time()
    test_loss, test_acc, test_precision, test_recall = model.evaluate(x_val, y_val, verbose=0)
    evaluation_time = time() - start_time

    print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
    print(f"Validation accuracy: {test_acc:.4f}")
    print(f"Validation loss: {test_loss:.4f}")
    print(f"Validation precision: {test_precision:.4f}")
    print(f"Validation recall: {test_recall:.4f}")

    # Generate prediction results
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Classification report
    print('\nClassification report:')
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES))

    # Plot the confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, CLASS_NAMES)

    # Calculate and display the accuracy of each class
    class_accuracies = {}
    for i in range(NUM_CLASSES):
        class_mask = (y_true_classes == i)
        class_acc = np.mean(y_pred_classes[class_mask] == i)
        class_accuracies[CLASS_NAMES[i]] = class_acc

    plt.figure(figsize=(10, 5))
    plt.bar(class_accuracies.keys(), class_accuracies.values())
    plt.title('Accuracy of each class', fontsize=12)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('class_accuracies.png')
    plt.show()


# Save prediction results to CSV
def save_predictions(model, x_test):
    """
    Save the model prediction results to a CSV file.

    Args:
        model: The trained model
        x_test: Test data
    """
    print("\nGenerating prediction results...")
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    test_files = os.listdir(os.path.join(DATA_DIR, 'test'))

    # Create a DataFrame containing the prediction results
    results_df = pd.DataFrame({
        'filename': test_files,
        'label': y_pred_classes
    })

    # Save to a CSV file
    results_df.to_csv(CSV_PATH, index=False)
    print(f"\nPrediction results saved to {CSV_PATH}")


# Main function
def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_val, y_val), x_test = load_and_preprocess_data()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    # Build the model
    print("\nBuilding the model...")
    model = build_cnn_model()
    model = compile_model(model)
    model.summary()

    # Visualize the model structure before training
    try:
        import pydot
        import graphviz
        tf.keras.utils.plot_model(model, to_file='model_architecture.png',
                                  show_shapes=True, show_layer_names=True)
        print("\nModel architecture diagram saved as model_architecture.png")
    except ImportError:
        print("\nUnable to generate the model architecture diagram. Please install pydot and graphviz.")
        print("You can install pydot using 'pip install pydot' and graphviz from its official website.")

    # Create callbacks
    callbacks_list = create_callbacks()

    # Train the model
    print("\nStarting model training...")
    start_time = time()
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    training_time = time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Visualize the training history
    plot_history(history)

    # Evaluate the model
    evaluate_model(model, x_val, y_val)

    # Save prediction results
    save_predictions(model, x_test)


if __name__ == "__main__":
    main()
    