import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from skimage import exposure
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
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

# Global variables to hold the current image and folder path
img_cv = None
image_folder = None
image_labels = []
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
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        processed_images.append(img)

    processed_images = np.array(processed_images) / 255.0  # Normalize images

    # Split into training and validation sets
    split_index = int(len(processed_images) * 0.8)
    train_images, val_images = processed_images[:split_index], processed_images[split_index:]

    # Create labels (dummy labels for demonstration)
    train_labels = np.array([0] * len(train_images))  # Assuming all images are "No Tumor" for training
    val_labels = np.array([0] * len(val_images))  # Assuming all images are "No Tumor" for validation

    # Train the model
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10)
    model.save('brain_tumor_detection_model.keras')
    messagebox.showinfo("Training Complete", "The model has been trained and saved successfully.")

def upload_folder():
    global image_folder, uploaded_image_paths, processed_images_cv
    image_folder = filedialog.askdirectory()
    if image_folder:
        display_images()  # Display images after uploading the folder

def display_images():
    global image_labels, uploaded_image_paths, processed_images_cv
    uploaded_image_paths = []  # Clear previous image paths
    processed_images_cv = []   # Clear previously processed images

    for label in image_labels:
        label.config(image='')  # Clear previous images

    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder, "*.png"))

    # Display up to 20 images
    for i, image_path in enumerate(image_paths[:20]):  # Limit to the first 20 images
        uploaded_image_paths.append(image_path)  # Store the path for predictions
        img = Image.open(image_path)
        img_cv = cv2.imread(image_path)  # Keep the OpenCV image for processing
        processed_images_cv.append(img_cv)
        
        img = img.resize((150, 150), Image.LANCZOS)  # Smaller size for display
        img_tk = ImageTk.PhotoImage(img)
        row = i // 5
        col = i % 5
        image_labels[i].config(image=img_tk)
        image_labels[i].image = img_tk  # Keep a reference to avoid garbage collection

def update_image_display():
    global processed_images_cv
    for index, img_cv in enumerate(processed_images_cv):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        img = img_pil.resize((150, 150), Image.LANCZOS)  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        image_labels[index].config(image=img_tk)
        image_labels[index].image = img_tk

# Image preprocessing functions (apply to all displayed images)
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
        M = np.float32([[1, 0.5, 0], [0, 1, 0]])  # Shear transformation
        processed_images_cv[i] = cv2.warpAffine(processed_images_cv[i], M, (cols, rows))
    update_image_display()

def random_cropping():
    global processed_images_cv
    for i in range(len(processed_images_cv)):
        h, w = processed_images_cv[i].shape[:2]
        x = np.random.randint(0, w // 4)
        y = np.random.randint(0, h // 4)
        processed_images_cv[i] = processed_images_cv[i][y:y + (3 * h // 4), x:x + (3 * w // 4)]
    update_image_display()

# Function to predict the result for a given image
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    
    if prediction < 0.5:
        return "No Tumor Detected"
    else:
        return "Tumor Detected"

# Display results for all images
def show_result():
    results = []
    for image_path in uploaded_image_paths:
        result = predict_image(image_path)
        results.append(f"{os.path.basename(image_path)}: {result}")
    result_label.config(text="\n".join(results))

# Main GUI setup
def main_gui(root):
    root.title("NeuroScan: Advanced Brain Tumor Detection System")
    root.geometry("1200x800")
    root.minsize(800, 600)

    frame = Frame(root, bg="white")
    frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=3)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=2)
    frame.grid_columnconfigure(2, weight=1)

    title = Label(frame, text="NeuroScan: Advanced Brain Tumor Detection System", font=("times new roman", 25, "bold"), bg="white")
    title.grid(row=0, column=0, columnspan=3, pady=20)

    # Create a button frame for image processing functions on the left side
    button_frame = Frame(frame, bg="white")
    button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ns")

    # Add buttons for data processing functions
    norm_button = Button(button_frame, text="Normalize", command=normalize_image, font=("times new roman", 12), bg="blue", fg="white")
    norm_button.pack(pady=5)

    noise_button = Button(button_frame, text="Noise Reduction", command=noise_reduction, font=("times new roman", 12), bg="blue", fg="white")
    noise_button.pack(pady=5)

    skull_button = Button(button_frame, text="Skull Stripping", command=skull_stripping, font=("times new roman", 12), bg="blue", fg="white")
    skull_button.pack(pady=5)

    artifact_button = Button(button_frame, text="Artifact Removal", command=artifact_removal, font=("times new roman", 12), bg="blue", fg="white")
    artifact_button.pack(pady=5)

    # Add buttons for data augmentation functions
    translate_button = Button(button_frame, text="Translate", command=translation, font=("times new roman", 12), bg="green", fg="white")
    translate_button.pack(pady=5)

    scale_button = Button(button_frame, text="Scale", command=scaling, font=("times new roman", 12), bg="green", fg="white")
    scale_button.pack(pady=5)

    flip_button = Button(button_frame, text="Flip", command=flipping, font=("times new roman", 12), bg="green", fg="white")
    flip_button.pack(pady=5)

    # Add buttons for advanced augmentation functions
    intensity_button = Button(button_frame, text="Intensity Adjustment", command=intensity_adjustment, font=("times new roman", 12), bg="purple", fg="white")
    intensity_button.pack(pady=5)

    shear_button = Button(button_frame, text="Shear", command=shearing, font=("times new roman", 12), bg="purple", fg="white")
    shear_button.pack(pady=5)

    crop_button = Button(button_frame, text="Random Crop", command=random_cropping, font=("times new roman", 12), bg="purple", fg="white")
    crop_button.pack(pady=5)

    train_button = Button(button_frame, text="Train Model", command=train_model, font=("times new roman", 12), bg="orange", fg="white")
    train_button.pack(pady=5)

    predict_button = Button(button_frame, text="Predict", command=show_result, font=("times new roman", 12), bg="red", fg="white")
    predict_button.pack(pady=5)

    upload_frame = Frame(frame, bg="white")
    upload_frame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

    upload_button = Button(upload_frame, text="Upload Dataset", command=upload_folder, font=("times new roman", 12), bg="orange", fg="white")
    upload_button.pack(pady=10)

    # Create a grid frame for displaying images
    image_display_frame = Frame(upload_frame, bg="white")
    image_display_frame.pack(fill="both", expand=True)

    # Create labels for displaying images in a grid (5x4 grid for 20 images)
    global image_labels
    image_labels = []
    for i in range(20):  # Increase to 20 labels for displaying images
        img_label = Label(image_display_frame, bg="lightgrey", width=150, height=150)  # Smaller size for better visibility
        img_label.grid(row=i // 5, column=i % 5, padx=5, pady=5)
        image_labels.append(img_label)

    global result_label
    result_label = Label(frame, text="", font=("times new roman", 18, "bold"), bg="white", fg="green")
    result_label.grid(row=2, column=1, columnspan=2, pady=20)

# Launch the application
root = tk.Tk()
main_gui(root)
root.mainloop()
