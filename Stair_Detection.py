import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_and_labels(stairs_dir, nonstair_dir):
    """Laad afbeeldingen en labels uit de opgegeven mappen voor trappen en geen trappen."""
    images = []
    labels = []
    image_names = []

    # Laad trappenafbeeldingen
    for image_name in os.listdir(stairs_dir):
        if image_name.endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(stairs_dir, image_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(1)  # Label 1 voor 'stair'
                image_names.append(image_name)

    # Laad geen trappenafbeeldingen
    for image_name in os.listdir(nonstair_dir):
        if image_name.endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(nonstair_dir, image_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(0)  # Label 0 voor 'nonstair'
                image_names.append(image_name)

    if not images:
        raise ValueError("Geen afbeeldingen gevonden in de opgegeven mappen.")

    images = np.array(images) / 255.0  # Normaliseer de afbeeldingen
    labels = np.array(labels)
    return images, labels, image_names

def build_improved_model():
    """Bouwt een lichter model met MobileNetV2 voor snellere training en transfer learning."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Vries de lagen van de basis MobileNetV2 model
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model(images, labels, batch_size=32, epochs=50):
    """Train het model met data-augmentatie en retourneert het getrainde model."""
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Print het aantal afbeeldingen in de testset
    print(f"Aantal afbeeldingen in de testset: {len(X_test)}")

    # Data-augmentatie
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.5],
    )
    datagen.fit(X_train)

    model = build_improved_model()

    # Callbacks voor vroegtijdig stoppen
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)

    # Plot de trainingsresultaten
    plt.figure(figsize=(12, 5))

    # Trainings- en validatieverlies
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Trainings- en validatie-accuraatheid
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    return model, X_test, y_test

def predict_and_show_all(model, images, labels, image_names):
    """Voorspel labels voor afbeeldingen en print de resultaten voor elke afbeelding."""
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(int)

    incorrect_predictions = 0
    results = []
    
    print("\nResultaten per afbeelding:")
    for i in range(len(images)):
        correct_label = "Trap" if labels[i] == 1 else "Geen Trap"
        predicted_label = "Trap" if predictions[i] == 1 else "Geen Trap"

        print(f"Afbeelding: {image_names[i]}, Correct Label: {correct_label}, Voorspelling: {predicted_label}")

        if predictions[i] != labels[i]:
            incorrect_predictions += 1
            results.append((images[i], image_names[i], correct_label, predicted_label))

    print(f"\nAantal foutieve voorspellingen: {incorrect_predictions} van de {len(images)} afbeeldingen.")

    if results:
        print("\nFoutieve voorspellingen:")
        for img, name, correct_label, predicted_label in results[:5]:  # Toon maximaal 5 foutieve voorspellingen
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{name} - Correct: {correct_label}, Voorspelling: {predicted_label}")
            plt.show()

def main():
    # Map voor trappen en geen trappen
    stairs_dir = "data/stairs"
    nonstair_dir = "data/non_stairs"

    try:
        # Laad afbeeldingen en labels
        images, labels, image_names = load_images_and_labels(stairs_dir, nonstair_dir)

        # Train het model
        model, X_test, y_test = train_model(images, labels)

        # Voorspel en toon de resultaten voor alle afbeeldingen
        predict_and_show_all(model, images, labels, image_names)

    except Exception as e:
        print(f"Fout: {e}")

if __name__ == "__main__":
    main()

