import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from skimage import exposure
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam

# Function to build the Convolutional Neural Network (CNN) model
def build_model():
    model = Sequential()
    
    # Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer to reduce spatial dimensions
    
    # Additional convolutional layers
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # Flattening layer to convert 2D matrix to 1D
    
    # Fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Initialize the CNN model
model = build_model()

# Uncomment this line to load a pre-trained model instead of building a new one
# model = load_model('brain_tumor_detection_model.keras')

# Function to train the model
def train_model():
    # Create an ImageDataGenerator for data augmentation and normalization
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, horizontal_flip=True, validation_split=0.2)

    # Generator for training data
    train_generator = train_datagen.flow_from_directory(
        'C:/Users/SOHAIL SONS/Downloads/BC220422733_CS619_Final+Deliverable/NeuroScan/images',  # Replace with your dataset path
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training')

    # Generator for validation data
    validation_generator = train_datagen.flow_from_directory(
        'C:/Users/SOHAIL SONS/Downloads/BC220422733_CS619_Final+Deliverable/NeuroScan/images',  # Replace with your dataset path
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation')

    # Train the model
    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    model.save('brain_tumor_detection_model.keras')  # Save the trained model
    messagebox.showinfo("Training Complete", "The model has been trained and saved successfully.")

# Function to upload an image from file
def upload_image():
    global img, img_tk, img_cv, file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = img.resize((600, 600), Image.LANCZOS)  # Resize image for display
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

# Function to update the image display in the GUI
def update_image_display():
    global img, img_tk
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img = img_pil.resize((600, 600), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Image processing functions

def normalize_image():
    global img_cv
    img_cv = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX)  # Normalize pixel values
    update_image_display()

def noise_reduction():
    global img_cv
    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)  # Reduce image noise
    update_image_display()

def skull_stripping():
    global img_cv
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]  # Apply thresholding
    img_cv = cv2.bitwise_and(img_cv, img_cv, mask=thresh)  # Apply mask to original image
    update_image_display()

def artifact_removal():
    global img_cv
    img_cv = cv2.medianBlur(img_cv, 5)  # Apply median blur to remove artifacts
    update_image_display()

# Data augmentation functions

def rotation():
    global img_cv
    (h, w) = img_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)  # Create rotation matrix
    img_cv = cv2.warpAffine(img_cv, M, (w, h))  # Apply rotation
    update_image_display()

def translation():
    global img_cv
    (h, w) = img_cv.shape[:2]
    M = np.float32([[1, 0, 25], [0, 1, 25]])  # Create translation matrix
    img_cv = cv2.warpAffine(img_cv, M, (w, h))  # Apply translation
    update_image_display()

def scaling():
    global img_cv
    img_cv = cv2.resize(img_cv, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)  # Scale image
    update_image_display()

def flipping():
    global img_cv
    img_cv = cv2.flip(img_cv, 1)  # Flip image horizontally
    update_image_display()

def intensity_adjustment():
    global img_cv
    img_cv = exposure.adjust_gamma(img_cv, gamma=0.4, gain=0.9)  # Adjust image intensity
    update_image_display()

def noise_injection():
    global img_cv
    row, col, ch = img_cv.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))  # Create Gaussian noise
    gauss = gauss.reshape(row, col, ch)
    img_cv = img_cv + gauss  # Add noise to the image
    update_image_display()

def shearing():
    global img_cv
    M = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  # Create shearing matrix
    img_cv = cv2.warpAffine(img_cv, M, (img_cv.shape[1], img_cv.shape[0]))  # Apply shearing
    update_image_display()

def random_cropping():
    global img_cv
    h, w = img_cv.shape[:2]
    x = np.random.randint(0, w // 2)  # Random x-coordinate
    y = np.random.randint(0, h // 2)  # Random y-coordinate
    img_cv = img_cv[y:y + h // 2, x:x + w // 2]  # Crop image randomly
    update_image_display()

# Function to predict whether the image has a tumor or not
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Preprocess image
    prediction = model.predict(img_array)  # Predict using the model
    
    if prediction < 0.5:
        return "No Tumor Detected"
    else:
        return "Tumor Detected"

# Function to display the result of the prediction
def show_result():
    result = predict_image(file_path)
    result_label.config(text=result)

# Function to set up the main GUI
def main_gui(root):
    root.title("NeuroScan: Advanced Brain Tumor Detection System")
    root.geometry("800x600")

    frame = Frame(root, bg="white")
    frame.grid(row=0, column=0, sticky="nsew")

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    frame.grid_rowconfigure(0, weight=0)  # Upload button
    frame.grid_rowconfigure(1, weight=0)  # Processing buttons
    frame.grid_rowconfigure(2, weight=0)  # Augmentation buttons
    frame.grid_rowconfigure(3, weight=0)  # Predict button
    frame.grid_rowconfigure(4, weight=1)  # Result display
    frame.grid_rowconfigure(5, weight=1)  # Image display

    frame.grid_columnconfigure(0, weight=1)

    upload_button = Button(frame, text="Upload Image", command=upload_image)
    upload_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    process_buttons_frame = Frame(frame, bg="white")
    process_buttons_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    Button(process_buttons_frame, text="Normalize", command=normalize_image).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    Button(process_buttons_frame, text="Noise Reduction", command=noise_reduction).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    Button(process_buttons_frame, text="Skull Stripping", command=skull_stripping).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
    Button(process_buttons_frame, text="Artifact Removal", command=artifact_removal).grid(row=0, column=3, padx=5, pady=5, sticky="ew")

    augment_buttons_frame = Frame(frame, bg="white")
    augment_buttons_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

    Button(augment_buttons_frame, text="Rotate", command=rotation).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    Button(augment_buttons_frame, text="Translate", command=translation).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    Button(augment_buttons_frame, text="Scale", command=scaling).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
    Button(augment_buttons_frame, text="Flip", command=flipping).grid(row=0, column=3, padx=5, pady=5, sticky="ew")
    Button(augment_buttons_frame, text="Adjust Intensity", command=intensity_adjustment).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
    Button(augment_buttons_frame, text="Inject Noise", command=noise_injection).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    Button(augment_buttons_frame, text="Shear", command=shearing).grid(row=1, column=2, padx=5, pady=5, sticky="ew")
    Button(augment_buttons_frame, text="Crop", command=random_cropping).grid(row=1, column=3, padx=5, pady=5, sticky="ew")

    result_label = Label(frame, text="", bg="white", font=("Arial", 16))
    result_label.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

    Button(frame, text="Predict", command=show_result).grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    global image_label
    image_label = Label(frame, bg="white")
    image_label.grid(row=5, column=0, padx=10, pady=10, sticky="nsew")

    frame.grid_rowconfigure(5, weight=1)
    frame.grid_columnconfigure(0, weight=1)

# Main Tkinter setup
if __name__ == "__main__":
    root = tk.Tk()
    main_gui(root)
    root.mainloop()
