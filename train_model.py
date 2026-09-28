import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def train():
    print("Loading MNIST dataset...")
    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add the channel dimension
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Data Augmentation layer to handle variations in real hand-drawn digits
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.05, input_shape=(28, 28, 1)), # small rotation (~18 degrees)
        layers.RandomTranslation(0.1, 0.1),                   # small shifts
        layers.RandomZoom(0.1)                                # slight zooming
    ], name="data_augmentation")

    # Define the CNN architecture
    model = models.Sequential([
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5), # prevent overfitting
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Early stopping to stop training if validation loss doesn't improve
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=2, 
        restore_best_weights=True
    )

    print("Starting training...")
    # Train for ~10 epochs
    model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=[early_stop])

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n=> Test accuracy: {test_acc:.4f}")

    # Save the trained model
    os.makedirs('model', exist_ok=True)
    model.save('model/mnist_cnn.keras')
    print("=> Model saved to model/mnist_cnn.keras")

if __name__ == '__main__':
    train()
