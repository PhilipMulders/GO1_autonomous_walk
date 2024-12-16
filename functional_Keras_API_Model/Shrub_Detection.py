import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

# Step 1: Load images and labels
def load_images_and_labels(image_dir, label_file):
    images = []
    labels = []
    image_names = []

    with open(label_file, 'r') as f:
        # Skip the header line
        header = next(f)
        print(f"Header: {header.strip()}")  # confirm the header is read and skipped

        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue

            label_str = parts[0].strip()  # "shrub"
            img_name = parts[1].strip()  # "frame_0.jpg"

            # Convert label to binary
            label = 1 if label_str == "shrub" else 0

            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue

                img = cv2.resize(img, (128, 128))  # Resize image
                images.append(img)
                labels.append(label)
                image_names.append(img_name)
            else:
                print(f"Image not found: {img_path}")
    
    # Convert to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Normalize images to [0, 1]
    images = images / 255.0

    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return images, labels, image_names


# Step 2: Build model tailored for shrubs
def build_model():
    # Define the input layer
    inputs = Input(shape=(128, 128, 3))  # 3 is for RGB
    # up from 64 to 128 to add more detail at cost of speed

    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second convolutional block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Third convolutional block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Fourth convolutional block (added for complexity of the shrubs)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)  # more neurons
    x = tf.keras.layers.Dropout(0.5)(x)   # Dropout to prevent overfitting
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification output

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Step 3: Train model with data augmentation and Early Stopping
def train_model(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True, # new
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    model = build_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test), callbacks=[early_stopping])


    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save the model after training
    model.save('model.h5')  # Save the trained model to a file called 'model.h5'
    print("Model saved as 'model.h5'")

    return model, X_test, y_test


def grad_cam(model, img, class_index, layer_name="conv2d"):
    class_index = int(class_index)

    grad_model = Model(
        inputs=model.input, 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    img_tensor = np.expand_dims(img, axis=0)  # Add batch dimension

    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]  # Class-specific loss

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = np.mean(conv_outputs * pooled_grads.numpy(), axis=-1)

    heatmap = np.maximum(heatmap, 0) # strict positive mask
    heatmap /= np.max(heatmap)
    return heatmap


# Visualization
def visualize_grad_cam(model, img, class_index, layer_name="conv2d"):
    heatmap = grad_cam(model, img, class_index, layer_name)
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Step 4: Predict and show Grad-CAM results
def predict_and_show(model, images, labels, image_names):
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype(int) # needs to be binary

    misclassified_results = []
    misclassified_images = []  # store misclassified imgs
    count = 0

    for i in range(len(images)):
        img = images[i]
        predicted_class = predictions[i][0]
        true_class = labels[i]
        
        label = "Shrub" if predicted_class == 1 else "No Shrub"
        true_label = "Shrub" if true_class == 1 else "No Shrub"

        # Only consider misclassified images
        if predicted_class != true_class:
            print(f"Misclassified: {image_names[i]} - Predicted: {label}, True: {true_label}")
            misclassified_results.append(f"Misclassified: {image_names[i]} - Predicted: {label}, True: {true_label}")
            misclassified_images.append(img)  # Store misclassified image for later use

            # Convert image and heatmap for visualization
            img_with_heatmap = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR)
            heatmap = grad_cam(model, img, predicted_class)
            heatmap_resized = cv2.resize(heatmap, (128, 128))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(img_with_heatmap, 0.6, heatmap_colored, 0.4, 0)

            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Image: {image_names[i]}, Predicted: {label}, True: {true_label}")
            plt.axis('off')
            plt.show()

            count += 1

            # Stop after 10 misclassified imgs
            if count >= 10:
                print("\nShowing only 10 misclassified imgs. Stopping display.")
                break

    # return misclassified images
    return misclassified_images


def main():
    image_dir = "data/frames"
    label_file = "data/labels.csv"

    images, labels, image_names = load_images_and_labels(image_dir, label_file)
    model, X_test, y_test = train_model(images, labels)

    predict_and_show(model, X_test, y_test, image_names)

if __name__ == "__main__":
    main()
