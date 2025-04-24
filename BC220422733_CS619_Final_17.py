import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from skimage import exposure
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (128, 128)
DISPLAY_SIZE = (150, 150)
LEARNING_RATE = 0.0001
MODEL_PATH = 'brain_tumor_detection_model.keras'
MAX_IMAGES_TO_DISPLAY = 20

# Load or build the model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize or load the model
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = build_model()
except Exception as e:
    messagebox.showerror("Model Load Error", f"Could not load the model: {e}")

# Global variables to hold the current image and folder paths
uploaded_image_paths = []  # Store paths of uploaded images for prediction
processed_images_cv = []   # Store processed images for display

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
    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, IMAGE_SIZE)
            processed_images.append(img)
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not process image {image_path}: {e}")

    processed_images = np.array(processed_images) / 255.0  # Normalize images

    # Split into training and validation sets
    split_index = int(len(processed_images) * 0.8)
    train_images, val_images = processed_images[:split_index], processed_images[split_index:]

    # Create dummy labels for demonstration
    train_labels = np.array([0] * len(train_images))  # Assuming all images are "No Tumor" for training
    val_labels = np.array([0] * len(val_images))      # Assuming all images are "No Tumor" for validation

    # Use EarlyStopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, callbacks=[early_stopping])
    model.save(MODEL_PATH)

    # Display training and validation accuracy
    messagebox.showinfo("Training Complete", f"Training Accuracy: {history.history['accuracy'][-1]:.4f}, Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

def upload_folder():
    global image_folder
    image_folder = filedialog.askdirectory()
    if image_folder:
        display_images()  # Display images after uploading the folder

def display_images():
    global uploaded_image_paths, processed_images_cv
    uploaded_image_paths.clear()  # Clear previous image paths
    processed_images_cv.clear()   # Clear previously processed images

    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder, "*.png"))

    # Display up to MAX_IMAGES_TO_DISPLAY images
    for i, image_path in enumerate(image_paths[:MAX_IMAGES_TO_DISPLAY]):  # Limit to the first 20 images
        uploaded_image_paths.append(image_path)  # Store the path for predictions
        try:
            img = Image.open(image_path)
            img_cv = cv2.imread(image_path)  # Keep the OpenCV image for processing
            processed_images_cv.append(img_cv)

            img = img.resize(DISPLAY_SIZE, Image.LANCZOS)  # Smaller size for display
            img_tk = ImageTk.PhotoImage(img)
            row = i // 5
            col = i % 5
            image_labels[i].config(image=img_tk)
            image_labels[i].image = img_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not display image {image_path}: {e}")

def update_image_display():
    global processed_images_cv
    for index, img_cv in enumerate(processed_images_cv):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        img = img_pil.resize(DISPLAY_SIZE, Image.LANCZOS)  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        image_labels[index].config(image=img_tk)
        image_labels[index].image = img_tk

# Image preprocessing functions
def normalize_image():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        processed_images_cv[i] = cv2.normalize(processed_images_cv[i], None, 0, 255, cv2.NORM_MINMAX)
    update_image_display()

def noise_reduction():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        processed_images_cv[i] = cv2.fastNlMeansDenoisingColored(processed_images_cv[i], None, 10, 10, 7, 21)
    update_image_display()

def skull_stripping():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        gray = cv2.cvtColor(processed_images_cv[i], cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
        processed_images_cv[i] = cv2.bitwise_and(processed_images_cv[i], processed_images_cv[i], mask=thresh)
    update_image_display()

def artifact_removal():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        processed_images_cv[i] = cv2.medianBlur(processed_images_cv[i], 5)
    update_image_display()

# Data Augmentation Functions
def translation():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        M = np.float32([[1, 0, 10], [0, 1, 10]])  # Translate 10 pixels to the right and down
        processed_images_cv[i] = cv2.warpAffine(processed_images_cv[i], M, (processed_images_cv[i].shape[1], processed_images_cv[i].shape[0]))
    update_image_display()

def scaling():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        processed_images_cv[i] = cv2.resize(processed_images_cv[i], None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
    update_image_display()

def flipping():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        processed_images_cv[i] = cv2.flip(processed_images_cv[i], 1)  # Flip horizontally
    update_image_display()

# Advanced Augmentation Functions
def intensity_adjustment():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        processed_images_cv[i] = cv2.convertScaleAbs(processed_images_cv[i], alpha=1.5, beta=0)  # Increase brightness
    update_image_display()

def shearing():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        rows, cols, _ = processed_images_cv[i].shape
        M = np.float32([[1, 0.2, 0], [0, 1, 0]])  # Shear matrix
        processed_images_cv[i] = cv2.warpAffine(processed_images_cv[i], M, (cols, rows))
    update_image_display()

def random_cropping():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        h, w, _ = processed_images_cv[i].shape
        x = np.random.randint(0, w // 4)
        y = np.random.randint(0, h // 4)
        processed_images_cv[i] = processed_images_cv[i][y:y + h // 2, x:x + w // 2]  # Crop to half the size
    update_image_display()

def noise_injection():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        noise = np.random.normal(0, 25, processed_images_cv[i].shape).astype(np.uint8)
        processed_images_cv[i] = cv2.add(processed_images_cv[i], noise)
    update_image_display()

# Tkinter UI Setup
root = tk.Tk()
root.title("NeuroScan: Advanced Brain Tumor Detection System")
root.geometry("800x600")

# Create frames for organization
frame_upload = Frame(root)
frame_upload.pack(pady=10)

frame_processing = Frame(root)
frame_processing.pack(pady=10)

frame_augmentation = Frame(root)
frame_augmentation.pack(pady=10)

# Upload folder button
upload_button = Button(frame_upload, text="Upload Image Folder", command=upload_folder)
upload_button.pack()

# Image display area with scroll
canvas = Canvas(frame_upload, width=600, height=200)
scrollbar = Scrollbar(frame_upload, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

image_labels = [Label(canvas) for _ in range(MAX_IMAGES_TO_DISPLAY)]
for label in image_labels:
    label.pack(side="top", padx=5, pady=5)

canvas.pack(side="left")
scrollbar.pack(side="right", fill="y")

# Buttons for image processing
Button(frame_processing, text="Normalize Image", command=normalize_image).pack(side="left", padx=5)
Button(frame_processing, text="Noise Reduction", command=noise_reduction).pack(side="left", padx=5)
Button(frame_processing, text="Skull Stripping", command=skull_stripping).pack(side="left", padx=5)
Button(frame_processing, text="Artifact Removal", command=artifact_removal).pack(side="left", padx=5)

# Buttons for data augmentation
Button(frame_augmentation, text="Translate", command=translation).pack(side="left", padx=5)
Button(frame_augmentation, text="Scale", command=scaling).pack(side="left", padx=5)
Button(frame_augmentation, text="Flip", command=flipping).pack(side="left", padx=5)
Button(frame_augmentation, text="Intensity Adjustment", command=intensity_adjustment).pack(side="left", padx=5)
Button(frame_augmentation, text="Shear", command=shearing).pack(side="left", padx=5)
Button(frame_augmentation, text="Random Crop", command=random_cropping).pack(side="left", padx=5)
Button(frame_augmentation, text="Noise Injection", command=noise_injection).pack(side="left", padx=5)

# Train Model button
train_button = Button(root, text="Train Model", command=train_model)
train_button.pack(pady=20)

root.mainloop()
