import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from skimage import exposure
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
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
model = build_model()  # Always build the model for this example

# Global variables to hold the current image and folder path
img_cv = None
current_image_path = None
image_folder = None

# Function to train the model
def train_model(image_folder):
    if not os.path.exists(image_folder):
        messagebox.showerror("Error", "Image folder does not exist.")
        return
    
    image_list = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                 glob.glob(os.path.join(image_folder, "*.jpeg")) + \
                 glob.glob(os.path.join(image_folder, "*.png"))

    if not image_list:
        messagebox.showerror("Error", "No images found in the selected folder.")
        return

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, horizontal_flip=True, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        image_folder,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training',
        shuffle=True)

    validation_generator = train_datagen.flow_from_directory(
        image_folder,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation',
        shuffle=True)

    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    model.save('brain_tumor_detection_model.keras')
    messagebox.showinfo("Training Complete", "The model has been trained and saved successfully.")

def upload_dataset():
    global image_folder
    image_folder = filedialog.askdirectory()
    if image_folder:
        display_images(image_folder)

def display_images(folder):
    global img_labels
    img_labels = []  # Reset image labels for new dataset
    for widget in image_display_frame.winfo_children():
        widget.destroy()  # Clear previous images

    image_list = glob.glob(os.path.join(folder, "*.jpg")) + \
                 glob.glob(os.path.join(folder, "*.jpeg")) + \
                 glob.glob(os.path.join(folder, "*.png"))

    if image_list:
        for index, image_path in enumerate(image_list):
            img = Image.open(image_path)
            img = img.resize((150, 150), Image.LANCZOS)  # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            img_label = Label(image_display_frame, image=img_tk)
            img_label.image = img_tk  # Keep a reference to avoid garbage collection
            img_label.grid(row=index // 5, column=index % 5, padx=5, pady=5)  # Create a grid layout
            img_labels.append(img_label)  # Store label references for potential future use
    else:
        messagebox.showerror("Error", "No images found in the selected folder.")

def update_image_display(image):
    global img_cv, current_image_path
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img_pil = img_pil.resize((600, 600), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Image processing functions
def normalize_image():
    global img_cv
    if img_cv is not None:
        img_cv = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX)
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for normalization.")

def noise_reduction():
    global img_cv
    if img_cv is not None:
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for noise reduction.")

def skull_stripping():
    global img_cv
    if img_cv is not None:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
        img_cv = cv2.bitwise_and(img_cv, img_cv, mask=thresh)
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for skull stripping.")

def artifact_removal():
    global img_cv
    if img_cv is not None:
        img_cv = cv2.medianBlur(img_cv, 5)
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for artifact removal.")

# Data augmentation functions
def rotation():
    global img_cv
    if img_cv is not None:
        (h, w) = img_cv.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)
        img_cv = cv2.warpAffine(img_cv, M, (w, h))
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for rotation.")

def translation():
    global img_cv
    if img_cv is not None:
        (h, w) = img_cv.shape[:2]
        M = np.float32([[1, 0, 25], [0, 1, 25]])
        img_cv = cv2.warpAffine(img_cv, M, (w, h))
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for translation.")

def scaling():
    global img_cv
    if img_cv is not None:
        img_cv = cv2.resize(img_cv, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for scaling.")

def flipping():
    global img_cv
    if img_cv is not None:
        img_cv = cv2.flip(img_cv, 1)
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for flipping.")

def intensity_adjustment():
    global img_cv
    if img_cv is not None:
        img_cv = exposure.adjust_gamma(img_cv, gamma=0.4, gain=0.9)
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for intensity adjustment.")

def noise_injection():
    global img_cv
    if img_cv is not None:
        row, col, ch = img_cv.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        img_cv = img_cv + gauss
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for noise injection.")

def shearing():
    global img_cv
    if img_cv is not None:
        M = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
        img_cv = cv2.warpAffine(img_cv, M, (img_cv.shape[1], img_cv.shape[0]))
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for shearing.")

def random_cropping():
    global img_cv
    if img_cv is not None:
        h, w = img_cv.shape[:2]
        x = np.random.randint(0, w // 2)
        y = np.random.randint(0, h // 2)
        img_cv = img_cv[y:y + h // 2, x:x + w // 2]
        update_image_display(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
    else:
        messagebox.showerror("Error", "No image loaded for random cropping.")

# Prediction function
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    
    if prediction < 0.5:
        return "No Tumor Detected"
    else:
        return "Tumor Detected"

# Display result
def show_result():
    global current_image_path
    if current_image_path:
        result = predict_image(current_image_path)
        result_label.config(text=result)
    else:
        messagebox.showerror("Error", "No image selected for prediction.")

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
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_columnconfigure(2, weight=1)

    title = Label(frame, text="NeuroScan: Advanced Brain Tumor Detection System", font=("times new roman", 25, "bold"), bg="white")
    title.grid(row=0, column=0, columnspan=3, pady=20)

    global image_display_frame
    image_display_frame = Frame(frame, bg="white")
    image_display_frame.grid(row=1, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")

    upload_button = Button(frame, text="Upload Dataset", command=upload_dataset, font=("times new roman", 12), bg="orange", fg="white")
    upload_button.grid(row=1, column=0, padx=10, pady=10)

    global image_label
    image_label = Label(frame, bg="lightgrey", width=600, height=600)
    image_label.grid(row=1, column=0, columnspan=2)

    button_frame = Frame(frame, bg="white")
    button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    process_title = Label(button_frame, text="Image Processing", font=("times new roman", 18, "bold"), bg="white")
    process_title.grid(row=0, column=0, pady=10)

    norm_button = Button(button_frame, text="Normalize", command=normalize_image, font=("times new roman", 12), bg="blue", fg="white")
    norm_button.grid(row=1, column=0, padx=10, pady=10)

    noise_button = Button(button_frame, text="Noise Reduction", command=noise_reduction, font=("times new roman", 12), bg="blue", fg="white")
    noise_button.grid(row=2, column=0, padx=10, pady=10)

    skull_button = Button(button_frame, text="Skull Stripping", command=skull_stripping, font=("times new roman", 12), bg="blue", fg="white")
    skull_button.grid(row=3, column=0, padx=10, pady=10)

    artifact_button = Button(button_frame, text="Artifact Removal", command=artifact_removal, font=("times new roman", 12), bg="blue", fg="white")
    artifact_button.grid(row=4, column=0, padx=10, pady=10)

    augment_title = Label(button_frame, text="Data Augmentation", font=("times new roman", 18, "bold"), bg="white")
    augment_title.grid(row=0, column=1, pady=10)

    rotate_button = Button(button_frame, text="Rotate", command=rotation, font=("times new roman", 12), bg="green", fg="white")
    rotate_button.grid(row=1, column=1, padx=10, pady=10)

    translate_button = Button(button_frame, text="Translate", command=translation, font=("times new roman", 12), bg="green", fg="white")
    translate_button.grid(row=2, column=1, padx=10, pady=10)

    scale_button = Button(button_frame, text="Scale", command=scaling, font=("times new roman", 12), bg="green", fg="white")
    scale_button.grid(row=3, column=1, padx=10, pady=10)

    flip_button = Button(button_frame, text="Flip", command=flipping, font=("times new roman", 12), bg="green", fg="white")
    flip_button.grid(row=4, column=1, padx=10, pady=10)

    augment2_title = Label(button_frame, text="Advanced Augmentation", font=("times new roman", 18, "bold"), bg="white")
    augment2_title.grid(row=0, column=2, pady=10)

    intensity_button = Button(button_frame, text="Intensity Adjustment", command=intensity_adjustment, font=("times new roman", 12), bg="purple", fg="white")
    intensity_button.grid(row=1, column=2, padx=10, pady=10)

    noise_inj_button = Button(button_frame, text="Noise Injection", command=noise_injection, font=("times new roman", 12), bg="purple", fg="white")
    noise_inj_button.grid(row=2, column=2, padx=10, pady=10)

    shear_button = Button(button_frame, text="Shearing", command=shearing, font=("times new roman", 12), bg="purple", fg="white")
    shear_button.grid(row=3, column=2, padx=10, pady=10)

    crop_button = Button(button_frame, text="Random Cropping", command=random_cropping, font=("times new roman", 12), bg="purple", fg="white")
    crop_button.grid(row=4, column=2, padx=10, pady=10)

    train_button = Button(button_frame, text="Train Model", command=lambda: train_model(image_folder), font=("times new roman", 12), bg="orange", fg="white")
    train_button.grid(row=5, column=0, padx=10, pady=10)

    predict_button = Button(button_frame, text="Predict", command=show_result, font=("times new roman", 12), bg="red", fg="white")
    predict_button.grid(row=5, column=1, padx=10, pady=10)

    global result_label
    result_label = Label(frame, text="", font=("times new roman", 18, "bold"), bg="white", fg="green")
    result_label.grid(row=2, column=0, columnspan=2, pady=20)

# Launch the application
root = tk.Tk()
main_gui(root)
root.mainloop()
