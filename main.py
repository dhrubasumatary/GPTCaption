# Importing necessary modules
import torch
import config
import openai
import textwrap
import streamlit as st
from huggingface_hub import notebook_login
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from tkinter import filedialog, Tk



# retrieve api_key from Streamlit secrets
secrets = st.secrets["api_keys"]
api_key = secrets["api_key"]

# set openai api_key
openai.api_key = api_key




# Loading pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU checked.")
else:
    device = torch.device("cpu")
    print("GPU not found, using CPU instead.")

# Moving the model to the device
model.to(device)

def generate_caption(image_file):
    try:
        # Open the image file and convert it to an RGB image
        image = Image.open(image_file).convert('RGB')
        # Converting the image to PyTorch tensor and moving it to the device
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        # Generating a caption for the image using the pre-trained model
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #return generated_caption
    #except Exception as e:
        #raise Exception("An error occurred while generating the caption:", e)

    # Generate text snippet using OpenAI API
        prompt = f"As an expert social media influencer, write three captions for this image on Instagram that will engage your audience:\n{generated_caption}\nInclude relevant hashtags and tag any relevant individuals or brands to increase engagement."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=4,
            stop=None,
            temperature=0,
        )
        # Extract generated text snippet from the OpenAI response
        generated_text = response.choices[0].text.strip()
        # Wrap the generated caption into multiple lines
        wrapped_text = textwrap.fill(generated_text, width=70)
        # Return the generated caption and text snippet
        return generated_caption, wrapped_text
    except Exception as e:
        raise Exception("An error occurred while generating the caption:", e)


def open_image():
    # Create a tkinter root window (required for file dialog)
    root = Tk()
    root.withdraw()  # Hide the root window
    # Show a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            # Open the image file and convert it to an RGB image
            image = Image.open(file_path).convert('RGB')
            print("Image opened successfully.")
            return image, file_path
        except Exception as e:
            raise Exception("An error occurred while opening the image file:", e)
    else:
        return None, None

def main():
    # Prompt user to select an image file
    image, file_path = open_image()
    if image:
        try:
            # Generate a caption for the selected image and print it to the console
            caption = generate_caption(image)
            print("Image file selected:", file_path)
            print("CAPTION 👉", caption)
            
            # Generate text snippet using OpenAI API
            prompt = f"As an expert social media influencer, write three caption for this image on Instagram that will engage your audience:\n{caption}\nInclude relevant hashtags and tag any relevant individuals or brands to increase engagement."

            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=100,
                n=4,
                stop=None,
                temperature=0,
            )

            # Extract generated text snippet from the OpenAI response
            generated_text = response.choices[0].text.strip()

            # Wrap the generated caption into multiple lines
            wrapped_text = textwrap.fill(generated_text, width=70)

            # Print the wrapped text
            print("\nGenerated caption:\n")
            print(wrapped_text)
            
        except Exception as e:
            print(e)

if __name__ == "__main__":
 main()