from tkinter import *
root = Tk()
root.title('My App')
root.mainloop()

import streamlit as st
from main import generate_caption

st.title('GPT Caption : Assignment Demo')
st.header('Automatically generate captions for your images!')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, use_column_width=True)

    # Generate caption
    #caption = generate_caption(uploaded_file)
    caption, text_snippet = generate_caption(uploaded_file)


    # Display caption
    st.subheader('Caption:')
    st.write(caption)

    # Display text snippet
    st.subheader('Text snippet for social media:')
    st.write(text_snippet)
