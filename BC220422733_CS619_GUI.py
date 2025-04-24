import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam

# Build the CNN model
def build_model():
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    # Compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Load the trained model (uncomment if loading an existing model)
# model = load_model('brain_tumor_detection_model.keras')

# Alternatively, you can train the model here
model = build_model()

# Function to train the model
def train_model():
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, horizontal_flip=True, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        'C:/Users/SOHAIL SONS/Downloads/BC220422733_CS619_Prototype/NeuroScan/images',  # Replace with your dataset path
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        # 'C:/Users/SOHAIL SONS/Downloads/BC220422733_CS619_Prototype/NeuroScan/images', 
        'C:/Users/SOHAIL SONS/Downloads/BC220422733_CS619_Final+Deliverable/NeuroScan/images',
        # Replace with your dataset path
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation')

    model.fit(train_generator, epochs=10, validation_data=validation_generator)
    model.save('brain_tumor_detection_model.keras')

# Function to predict the presence of a tumor in the image
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    
    if prediction < 0.5:
        return "No Tumor Detected"
    else:
        return "Tumor Detected"

# Tkinter GUI code
class TumorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('NeuroScan: Advanced Brain Tumor Detection System')

        # Upload Button
        self.upload_btn = tk.Button(root, text='Upload Image', command=self.upload_image)
        self.upload_btn.pack()

        # Image Display
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Result Display
        self.result_label = tk.Label(root, text='', font=('Arial', 14))
        self.result_label.pack()

        # Prediction Button
        self.predict_btn = tk.Button(root, text='Predict Tumor', command=self.show_result)
        self.predict_btn.pack()

        # Train Button
        self.train_btn = tk.Button(root, text='Train Model', command=train_model)
        self.train_btn.pack()

    def upload_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            img = Image.open(self.image_path)
            img = img.resize((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

    def show_result(self):
        if hasattr(self, 'image_path'):
            result = predict_image(self.image_path)
            self.result_label.config(text=result)
        else:
            messagebox.showerror("Error", "Please upload an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDetectionApp(root)
    root.mainloop()
