# Import necessary libraries
import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from skimage import exposure
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import glob

# Load or build the model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize or load the model
if os.path.exists('brain_tumor_detection_model.keras'):
    model = load_model('brain_tumor_detection_model.keras')
else:
    model = build_model()

# Global variables to hold the current image and folder paths
img_cv = None
image_folder = None
image_labels = []
uploaded_image_paths = []  # Store paths of uploaded images for prediction
processed_images_cv = []   # Store processed images for display

# Variables for Test Dataset
test_image_paths = []
test_labels = []
test_predictions = []

# Function to augment the dataset and split into train/test sets
def augment_and_split_data(images, labels):
    # Multiply the dataset threefold using data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        img = img.reshape((1,) + img.shape)  # Reshape for the ImageDataGenerator
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            i += 1
            if i >= 3:  # Augment each image 3 times
                break
    
    # Combine original and augmented data
    images = np.concatenate((images, np.array(augmented_images)))
    labels = np.concatenate((labels, np.array(augmented_labels)))
    
    # Split into 80% train and 20% test sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    return train_images, test_images, train_labels, test_labels

# Function to train the model
def train_model():
    global image_folder
    # Preprocess images
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder, "*.png"))

    if not image_paths:
        messagebox.showerror("Error", "No images found in the selected folder.")
        return

    processed_images = []
    labels = []  # Dummy labels (replace with actual labels)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        processed_images.append(img)

        # For now, assigning dummy labels (0 = No Tumor, 1 = Tumor)
        labels.append(0 if "no_tumor" in image_path.lower() else 1)

    processed_images = np.array(processed_images) / 255.0  # Normalize images
    labels = np.array(labels)

    # Augment and split the data
    train_images, test_images, train_labels, test_labels = augment_and_split_data(processed_images, labels)

    # Train the model
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10)
    model.save('brain_tumor_detection_model.keras')
    messagebox.showinfo("Training Complete", "The model has been trained and saved successfully.")

# Function to predict the result for a given image
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    
    if prediction < 0.5:
        return 0  # No Tumor Detected
    else:
        return 1  # Tumor Detected

# Display results for all images
def show_result():
    results = []
    for image_path in uploaded_image_paths:
        result = predict_image(image_path)
        results.append(f"{os.path.basename(image_path)}: {'Tumor Detected' if result == 1 else 'No Tumor Detected'}")
    result_label.config(text="\n".join(results))

# Function to upload the folder
def upload_folder():
    global image_folder, uploaded_image_paths, processed_images_cv
    image_folder = filedialog.askdirectory()
    if image_folder:
        display_images()  # Display images after uploading the folder

# Function to reset the data
def reset():
    global uploaded_image_paths, processed_images_cv, test_image_paths, test_labels, test_predictions
    uploaded_image_paths = []
    processed_images_cv = []
    test_image_paths = []
    test_labels = []
    test_predictions = []
    for label in image_labels:
        label.config(image='')  # Clear image display
    result_label.config(text="")  # Clear results

# Main GUI setup
def main_gui(root):
    root.title("NeuroScan: Advanced Brain Tumor Detection System")
    root.geometry("1400x900")  # Increased size to accommodate additional buttons
    root.minsize(800, 600)

    frame = Frame(root, bg="white")
    frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    title = Label(frame, text="NeuroScan: Advanced Brain Tumor Detection System", font=("times new roman", 25, "bold"), bg="white")
    title.grid(row=0, column=0, columnspan=3, pady=20)

    # Create a button frame for image processing functions on the left side
    button_frame = Frame(frame, bg="white")
    button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ns")

    # Train Model Button
    train_button = Button(button_frame, text="Train Model", command=train_model, font=("times new roman", 12), bg="orange", fg="white")
    train_button.pack(pady=5)

    # Reset Button
    reset_button = Button(button_frame, text="Reset", command=reset, font=("times new roman", 12), bg="red", fg="white")
    reset_button.pack(pady=5)

    # Upload Dataset Button
    upload_button = Button(button_frame, text="Upload Dataset", command=upload_folder, font=("times new roman", 12), bg="cyan", fg="black")
    upload_button.pack(pady=5)

    # Predict Button
    predict_button = Button(button_frame, text="Predict", command=show_result, font=("times new roman", 12), bg="red", fg="white")
    predict_button.pack(pady=5)

    global result_label
    result_label = Label(frame, text="", font=("times new roman", 18, "bold"), bg="white", fg="green")
    result_label.grid(row=2, column=1, columnspan=2, pady=20)

# Launch the application
if __name__ == "__main__":
    root = tk.Tk()
    main_gui(root)
    root.mainloop()
