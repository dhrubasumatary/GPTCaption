# Importing necessary modules
from huggingface_hub import notebook_login
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import matplotlib.pyplot as plt
import requests
import openai
from tkinter import filedialog
from tkinter import Tk


# Loading pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Print types of model, feature_extractor, and tokenizer
print(type(model))
print(type(feature_extractor))
print(type(tokenizer))



# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU checked.")
else:
    device = torch.device("cpu")
    print("GPU not found, using CPU instead.")


# Moving the model to the device
model.to(device)


def generate_caption(image_path):

    # Open the image file and convert it to an RGB image
    try:
        image = Image.open(image_path).convert('RGB')
        print("Image opened successfully.")

    except Exception as e:
        print("An error occurred while opening the image file:", e)


    # Converting the image to PyTorch tensor and moving it to the device
    try:
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        print("Image converted to PyTorch tensor and moved to device.")
    except Exception as e:
        print("An error occurred while converting the image to PyTorch tensor:", e)


    # Generating a caption for the image using the pre-trained model
    try:
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Caption generated successfully.")
    except Exception as e:
        print("An error occurred while generating the caption:", e)

    # Returning the generated caption
    return generated_caption



# Create a tkinter root window (required for file dialog)
root = Tk()
root.withdraw()  # Hide the root window

# Show a file dialog to select an image file
file_path = filedialog.askopenfilename(
    initialdir="/home/xai/Desktop",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
)
print("Image file selected:", file_path)

# Open the image file and convert it to an RGB image
try:
        image = Image.open(file_path).convert('RGB')
        print("Image opened successfully.")

except Exception as e:
        print("An error occurred while opening the image file:", e)

# Create a PyTorch tensor from the image data and reshape it to match the size of the input image
tensor = torch.Tensor(list(image.getdata())).view(image.size[1], image.size[0], 3)

# Unsqueeze the tensor along the 0th dimension to create a batch of size 1
tensor = tensor.unsqueeze(0)

# Call the `generate_caption()` function with the `file_path` as input and assign the generated caption to the variable `caption`
caption = generate_caption(file_path)

# Print the generated caption to the console
print("CAPTION 👉", caption)
