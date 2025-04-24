import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import glob

# Initialize the main window
root = tk.Tk()
root.title("Image Processing with Horizontal Scrollbar")
root.geometry("800x600")

# Create a Canvas widget with a horizontal scrollbar
canvas = Canvas(root)
scroll_x = Scrollbar(root, orient="horizontal", command=canvas.xview)
canvas.configure(xscrollcommand=scroll_x.set)

# Create a frame to hold the images (inside the Canvas)
frame = Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

# Function to update the scrollbar region
def update_scroll_region():
    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

# Label widgets to display images (limited to 20 as per your original code)
image_labels = [Label(frame) for _ in range(20)]
for label in image_labels:
    label.grid(padx=5, pady=5)

# Example function to upload images
def upload_folder():
    global image_folder, uploaded_image_paths, processed_images_cv
    image_folder = filedialog.askdirectory()
    if image_folder:
        display_images()  # Display images after uploading the folder

# Function to display the images in the scrollable frame
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
        image_labels[i].config(image=img_tk)
        image_labels[i].image = img_tk  # Keep a reference to avoid garbage collection

    update_scroll_region()  # Update the scrollbar's scrollable region

# Upload button
upload_button = Button(root, text="Upload Folder", command=upload_folder)
upload_button.pack(side="top", pady=10)

# Pack the canvas and scrollbar at the bottom
canvas.pack(side="top", fill="both", expand=True)
scroll_x.pack(side="bottom", fill="x")

root.mainloop()
