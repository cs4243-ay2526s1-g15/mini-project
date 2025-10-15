import os
import shutil
from tkinter import Tk, Button, Label, PhotoImage
from PIL import Image, ImageTk
from tkinter.simpledialog import askstring

# Path to your images
IMAGE_FOLDER = r"dataset\test\processed_failed"
KEEP_FOLDER = r"dataset\test\processed_success"
DELETE_FOLDER = r"dataset\test\processed_failed"

# Make sure folders exist
os.makedirs(KEEP_FOLDER, exist_ok=True)
os.makedirs(DELETE_FOLDER, exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png"))]
index = 0

# Function to show next image
def show_image():
    global index, img_label, filename_label, tk_image

    if index >= len(image_files):
        print("No more images!")
        root.destroy()
        return

    file_path = os.path.join(IMAGE_FOLDER, image_files[index])
    pil_image = Image.open(file_path)

    # Resize if too big
    pil_image.thumbnail((800, 800))
    tk_image = ImageTk.PhotoImage(pil_image)

    img_label.config(image=tk_image)
    filename_label.config(text=image_files[index])

# Button callbacks
def keep():
    global index
    shutil.move(os.path.join(IMAGE_FOLDER, image_files[index]), os.path.join(KEEP_FOLDER, image_files[index]))
    index += 1
    show_image()

def delete():
    global index
    #shutil.move(os.path.join(IMAGE_FOLDER, image_files[index]), os.path.join(DELETE_FOLDER, image_files[index]))
    index += 1
    show_image()

def rename():
    global index
    
    if index >= len(image_files):
        return
    
    old_path = os.path.join(IMAGE_FOLDER, image_files[index])
    # Ask user for new name
    new_name = askstring("Rename Image", f"Enter new name without extension:")
    if new_name:  # Only rename if user entered something
        new_name = f"{new_name}-0.png"
        new_path = os.path.join(KEEP_FOLDER, new_name)
        os.rename(old_path, new_path)
        image_files[index] = new_name  # update list for next display
    
    index += 1
    show_image()
# GUI setup
root = Tk()
root.title("Image Reviewer")

img_label = Label(root)
img_label.pack()

filename_label = Label(root, text="", font=("Arial", 14))
filename_label.pack()

btn_keep = Button(root, text="Success", command=keep, width=10, bg="green", fg="white")
btn_keep.pack(side="left", padx=10, pady=10)

btn_delete = Button(root, text="Fail", command=delete, width=10, bg="red", fg="white")
btn_delete.pack(side="right", padx=10, pady=10)

btn_rename = Button(root, text = "Rename", command=rename, width=10, bg="blue", fg="white")
btn_rename.pack(side="bottom", padx=10, pady=10)

show_image()
root.mainloop()
