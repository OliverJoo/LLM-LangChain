import base64
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import requests
import os
from openai import OpenAI, BadRequestError
import logging
import time

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide")
st.title("Document OCR and Summarization")

# --- OpenAI API Client ---
# Initialize the OpenAI client with API key from environment variables.
try:
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    st.error("Failed to initialize OpenAI client. Please check your API key.")
    exit()

# --- Language Mapping ---
language_mapping = {
    "English": "en",
    "한국어": "ko",
    "日本語": "ja",
    "中文": "zh",
    "Italiano": "it",
    "Ελληνικά": "el",
    "עברית": "he"
}

def ocr_with_openai_vision(image_bytes, target_language):
    """
    Performs OCR processing on an image using OpenAI's vision API.

    Args:
        image_bytes (bytes): The image data as bytes.
        target_language (str): The language code for the target language (e.g., "ko", "en").

    Returns:
        str: The extracted text from the image in the target language, or None if an error occurred.
    """
    logger.info(f"Starting OCR processing with OpenAI's vision API to {target_language}...")
    try:
        base64_image = base64_encode(image_bytes)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"이 이미지에 어떤 내용이 있는지 {get_language_name(target_language)}로 설명해줘."}, # Modified : change to dynamic language
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        if response and response.choices and response.choices[0].message.content:
            extracted_text = response.choices[0].message.content
            logger.info(f"OCR processing completed successfully in {target_language}.")
            return extracted_text
        else:
            logger.error("OCR processing did not return valid text.")
            st.error("OCR processing did not return valid text.")
            return None
    except BadRequestError as e:
        logger.error(f"BadRequestError during OCR processing: {e}")
        st.error(f"BadRequestError occurred: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}")
        st.error(f"An unexpected error occurred during OCR processing: {e}")
        return None

def get_language_name(language_code: str) -> str:
    """
    Get the name of the language corresponding to the language code.
    """
    for name, code in language_mapping.items():
        if code == language_code:
            return name
    return "unknown"


def base64_encode(image_bytes):
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")

def summarize_text(text):
    """
    Summarizes the given text into bullet points in Korean using OpenAI's ChatCompletion API.

    Args:
        text (str): The text to summarize.

    Returns:
        str: The summarized text in Korean, or None if an error occurred.
    """
    logger.info("Starting text summarization...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text into bullet points in Korean."}, # Modified : add "in Korean"
                {"role": "user", "content": text}
            ],
            temperature=0.5,
        )
        if response and response.choices and response.choices[0].message.content:
            summarized_text = response.choices[0].message.content
            logger.info("Text summarization completed successfully.")
            return summarized_text
        else:
            logger.error("Text summarization did not return valid text.")
            st.error("Text summarization did not return valid text.")
            return None
    except BadRequestError as e:
        logger.error(f"BadRequestError during text summarization: {e}")
        st.error(f"BadRequestError occurred: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during text summarization: {e}")
        st.error(f"An unexpected error occurred during text summarization: {e}")
        return None

def display_ocr_results(image, text):
    """Displays the processed image and the extracted text in Streamlit."""
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Processed Image', use_column_width=True)
        with col2:
            st.write("Text in image:")
            st.code(text)
    except Exception as e:
        logger.error(f"Error displaying OCR results: {e}")
        st.error(f"An unexpected error occurred displaying OCR results: {e}")

# --- File Uploader ---
uploaded_file = st.file_uploader("이미지를 업로드하세요:", type=["png", "jpg", "jpeg"])
base64_image = None
if uploaded_file is not None:
    # Read the content of the file as bytes
    image_bytes = uploaded_file.getvalue()
    # Language selection dropdown
    selected_language = st.selectbox("번역할 언어를 선택하세요", list(language_mapping.keys()))

    # Button to trigger OCR processing
    if st.button("이미지 읽기!"):
        start_time = time.time()
        if image_bytes:
            ocr_result = ocr_with_openai_vision(image_bytes, language_mapping[selected_language]) # Modified : add language mapping
            if ocr_result:
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f"OCR completed in {elapsed_time:.2f} seconds")
                st.session_state['ocr_result'] = ocr_result
                st.session_state['original_image'] = Image.open(uploaded_file)  # Save original image
                st.session_state['ocr_clicked'] = True
            else:
                st.error("OCR processing failed. Please try again.")
        else:
            st.error("No image data available for processing.")

if 'ocr_result' in st.session_state and 'original_image' in st.session_state:
    display_ocr_results(st.session_state['original_image'], st.session_state['ocr_result'])

    # Extracted text from OCR
    extracted_text = st.session_state['ocr_result'] if 'ocr_result' in st.session_state else ""

    if st.button("요약하기"):
        summary = summarize_text(extracted_text)
        if summary:
            st.write("요약:")
            st.write(summary)
        else:
            st.error("요약을 생성하는데 실패했습니다.")