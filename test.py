from tkinter import filedialog
from tkinter import Tk
from PIL import Image

# Create a tkinter root window (required for file dialog)
root = Tk()
root.withdraw()  # Hide the root window

# Show a file dialog to select an image file
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Load the selected image file using PIL
image = Image.open(file_path)

# Perform image captioning using the `generate_caption()` function
caption = generate_caption(image)

# Print the generated caption to the console
print("CAPTION 👉", caption)
