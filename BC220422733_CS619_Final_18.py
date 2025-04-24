import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, messagebox, Scrollbar, Canvasfrom tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Reset function to clear images and results
def reset():
    global uploaded_image_paths
    uploaded_image_paths = []
    for label in image_labels:
        label.config(image='')  # Clear image display
    result_label.config(text="")  # Clear bulk results

# Function to upload dataset
def upload_folder():
    global image_folder, uploaded_image_paths
    image_folder = filedialog.askdirectory()
    if image_folder:
        display_images()

# Function to display images
def display_images():
    global uploaded_image_paths
    uploaded_image_paths = []
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder, "*.png"))
    
    for i, image_path in enumerate(image_paths[:20]):
        uploaded_image_paths.append(image_path)
        img = Image.open(image_path)
        img = img.resize((150, 150), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        row = i // 5
        col = i % 5
        image_labels[i].config(image=img_tk)
        image_labels[i].image = img_tk

# Main GUI setup
def main_gui(root):
    root.title("NeuroScan: Advanced Brain Tumor Detection System")
    root.geometry("1600x1000")  # Increased size to accommodate additional buttons and display areas
    root.minsize(800, 600)

    # Create a frame for the canvas
    canvas = Canvas(root)
    scroll_y = Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_x = Scrollbar(root, orient="horizontal", command=canvas.xview)

    scrollable_frame = Frame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Configure scrollbar
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")

    frame = Frame(scrollable_frame, bg="white")
    frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

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

    # Data Processing Functions
    noise_button = Button(button_frame, text="Noise Reduction", font=("times new roman", 12), bg="blue", fg="white")
    noise_button.pack(pady=5)

    skull_button = Button(button_frame, text="Skull Stripping", font=("times new roman", 12), bg="blue", fg="white")
    skull_button.pack(pady=5)

    artifact_button = Button(button_frame, text="Artifact Removal", font=("times new roman", 12), bg="blue", fg="white")
    artifact_button.pack(pady=5)

    # Train Model Button
    train_button = Button(button_frame, text="Train Model", font=("times new roman", 12), bg="orange", fg="white")
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

    # Create a grid frame for displaying images
    image_display_frame = Frame(frame, bg="white")
    image_display_frame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

    # Create labels for displaying images in a grid (5x4 grid for 20 images)
    global image_labels
    image_labels = []
    for i in range(20):
        img_label = Label(image_display_frame, bg="lightgrey", width=150, height=150)
        img_label.grid(row=i // 5, column=i % 5, padx=5, pady=5)
        image_labels.append(img_label)

    global result_label
    result_label = Label(frame, text="", font=("times new roman", 18, "bold"), bg="white", fg="green")
    result_label.grid(row=2, column=1, columnspan=2, pady=20)

# Launch the application
if __name__ == "__main__":
    root = tk.Tk()
    main_gui(root)
    root.mainloop()
