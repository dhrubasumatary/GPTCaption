# Importing necessary modules
from huggingface_hub import notebook_login
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import matplotlib.pyplot as plt
import requests
import openai


# Loading pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Print types of model, feature_extractor, and tokenizer
print(type(model))
print(type(feature_extractor))
print(type(tokenizer))


# Load ViTImageProcessor to convert image to PyTorch tensor
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
#print(feature_extractor)


# Setting the device to use cuda if it is available, otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Moving the model to the device
model.to(device)


def generate_caption(image):
    # Converting the image to PyTorch tensor and moving it to the device
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generating a caption for the image using the pre-trained model
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Returning the generated caption
    return generated_caption


# Open the image file and convert it to an RGB image
image = Image.open('/home/xai/Lab/XD/transformers/Image3-min.png')
image = image.convert('RGB')

# Create a PyTorch tensor from the image data and reshape it to match the size of the input image
tensor = torch.Tensor(list(image.getdata())).view(image.size[1], image.size[0], 3)

# Unsqueeze the tensor along the 0th dimension to create a batch of size 1
tensor = tensor.unsqueeze(0)

# Call the `generate_caption()` function with the `image` tensor as input and assign the generated caption to the variable `caption`
caption = generate_caption(image)

# Print the generated caption to the console
print("CAPTION 👉", caption)
