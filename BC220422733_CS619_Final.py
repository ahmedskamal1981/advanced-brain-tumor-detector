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

# Function to train the model
def train_model():
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, horizontal_flip=True, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        '/home/ahmed/Downloads/BC220422733_CS619_Final+Deliverable/NeuroScan/images',  # Replace with your dataset path
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        '/home/ahmed/Downloads/BC220422733_CS619_Final+Deliverable/NeuroScan/images',  # Replace with your dataset path
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation')

    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    model.save('brain_tumor_detection_model.keras')
    messagebox.showinfo("Training Complete", "The model has been trained and saved successfully.")

def upload_image():
    global img, img_tk, img_cv, file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = img.resize((600, 600), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

def update_image_display():
    global img, img_tk
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img = img_pil.resize((600, 600), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Image preprocessing functions
def normalize_image():
    global img_cv
    img_cv = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX)
    update_image_display()

def noise_reduction():
    global img_cv
    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    update_image_display()

def skull_stripping():
    global img_cv
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    img_cv = cv2.bitwise_and(img_cv, img_cv, mask=thresh)
    update_image_display()

def artifact_removal():
    global img_cv
    img_cv = cv2.medianBlur(img_cv, 5)
    update_image_display()

# Data augmentation functions
def rotation():
    global img_cv
    (h, w) = img_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_cv = cv2.warpAffine(img_cv, M, (w, h))
    update_image_display()

def translation():
    global img_cv
    (h, w) = img_cv.shape[:2]
    M = np.float32([[1, 0, 25], [0, 1, 25]])
    img_cv = cv2.warpAffine(img_cv, M, (w, h))
    update_image_display()

def scaling():
    global img_cv
    img_cv = cv2.resize(img_cv, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    update_image_display()

def flipping():
    global img_cv
    img_cv = cv2.flip(img_cv, 1)
    update_image_display()

def intensity_adjustment():
    global img_cv
    img_cv = exposure.adjust_gamma(img_cv, gamma=0.4, gain=0.9)
    update_image_display()

def noise_injection():
    global img_cv
    row, col, ch = img_cv.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    img_cv = img_cv + gauss
    update_image_display()

def shearing():
    global img_cv
    M = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
    img_cv = cv2.warpAffine(img_cv, M, (img_cv.shape[1], img_cv.shape[0]))
    update_image_display()

def random_cropping():
    global img_cv
    h, w = img_cv.shape[:2]
    x = np.random.randint(0, w // 2)
    y = np.random.randint(0, h // 2)
    img_cv = img_cv[y:y + h // 2, x:x + w // 2]
    update_image_display()

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
    result = predict_image(file_path)
    result_label.config(text=result)

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

    upload_frame = Frame(frame, bg="white")
    upload_frame.grid(row=1, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")

    upload_button = Button(upload_frame, text="Upload Image", command=upload_image, font=("times new roman", 12), bg="orange", fg="white")
    upload_button.pack(pady=10)

    global image_label
    image_label = Label(upload_frame, bg="lightgrey", width=600, height=600)
    image_label.pack()

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

    noise_button = Button(button_frame, text="Noise Injection", command=noise_injection, font=("times new roman", 12), bg="purple", fg="white")
    noise_button.grid(row=2, column=2, padx=10, pady=10)

    shear_button = Button(button_frame, text="Shearing", command=shearing, font=("times new roman", 12), bg="purple", fg="white")
    shear_button.grid(row=3, column=2, padx=10, pady=10)

    crop_button = Button(button_frame, text="Random Cropping", command=random_cropping, font=("times new roman", 12), bg="purple", fg="white")
    crop_button.grid(row=4, column=2, padx=10, pady=10)

    train_button = Button(button_frame, text="Train Model", command=train_model, font=("times new roman", 12), bg="orange", fg="white")
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
  
