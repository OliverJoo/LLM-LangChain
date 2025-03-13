import streamlit as st
from openai import AsyncOpenAI, OpenAI
import asyncio
import pandas as pd
from io import BytesIO
import os

# Default model to be used for correction
default_model = "gpt-4o-mini"

# Set page configuration for wide layout
st.set_page_config(layout="wide")

# Title of the app
st.title("Multi-Model Sentence Corrector")
st.text(f"Correct sentences using OpenAI's default model ({default_model}) and user-specified fine-tuned models.")

# Initialize OpenAI clients for asynchronous and synchronous operations
async_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Split the layout into two columns
col1, col2 = st.columns([0.3, 0.7])


async def correct(text, model, model_num):
    """
    Corrects a given text using the specified OpenAI model.

    Args:
        text (str): The text to be corrected.
        model (str): The OpenAI model to use.
        model_num (int): The index of the model (used for model alias).
    """
    messages = [
        {"role": "system", "content": "Please correct the following sentence into natural language."},
        {"role": "user", "content": text}
    ]
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0  # Set temperature to 0.0 for deterministic output
    )
    corrected_text = response.choices[0].message.content.strip()
    # Determine the model alias for display
    model_alias = f"Default Model" if model == default_model \
        else f"Fine-tuned Model {model_num}" if model.startswith("ft:") \
        else f"Custom Model {model_num}"
    st.session_state[model_alias] = corrected_text


async def main():
    """
    Main asynchronous function to handle text correction with multiple models.
    """
    if text.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        # Combine the default model with any user-specified fine-tuned models
        models = [default_model] + finetuned_models
        # Create tasks for each model's correction process
        tasks = [asyncio.create_task(correct(text, model, i)) \
                 for i, model in enumerate(models)
                 ]
        # Wait for all correction tasks to complete
        await asyncio.gather(*tasks)
        # Initialize the best_sentence in the session_state
        if "best_sentence" not in st.session_state:
            st.session_state["best_sentence"] = ""


with col1:
    # Initialize session state for the input text if it does not exist
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    # Text area for the user to input the sentence to be corrected
    text = st.text_area(
        "Enter the sentence to be corrected:", height=400, key="input_text",
        value=st.session_state["input_text"]
    )

    # Input field for additional fine-tuned models, separated by commas
    finetuned_models = st.text_input(
        "Enter additional models to use (comma-separated):", ""
    )
    # Split and clean the input models
    finetuned_models = [
        model.strip() for model in finetuned_models.split(",") if model.strip()
    ]

    # Button to trigger the correction process
    if st.button("Correct"):
        asyncio.run(main())

with col2:
    # Display correction results for each model
    for i, model in enumerate([default_model] + finetuned_models):
        if model == default_model:
            model_alias = f"Default Model"
        elif model.startswith("ft:"):
            model_alias = f"Fine-tuned Model {i}"
        else:
            model_alias = f"Custom Model {i}"

        # Display the corrected sentence from each model
        if model_alias in st.session_state:
            # Select the best sentence
            if st.button(model_alias, key=f"select_{model_alias}"):
                st.session_state["best_sentence"] = st.session_state[model_alias]
            st.text_area(
                model_alias, st.session_state[model_alias], height=180,
                key=f"display_{model_alias}"
            )

# Initialize session state for the best sentence if it does not exist
if "best_sentence" not in st.session_state:
    st.session_state["best_sentence"] = ""

# Text area to select or input the best corrected sentence
best_sentence = st.text_area(
    "Enter the best sentence or select from above:",
    height=200,
    key="best_sentence"
)

# Button to trigger text-to-speech (TTS)
if st.button("Listen with TTS"):
    response = client.audio.speech.create(
        model="tts-1", voice="alloy", input=best_sentence
    )
    # Audio data to BytesIO for Streamlit audio component
    audio_data = BytesIO(response.read())
    st.audio(audio_data, format='audio/mpeg')

# Initialize session state for sentences if it does not exist
if "sentences" not in st.session_state:
    st.session_state.sentences = []

# Button to save the original and corrected sentence pair
if st.button("Save"):
    st.session_state.sentences.append({"original": text, "corrected": best_sentence})
    st.success("The sentence pair has been saved.")

    # Create a DataFrame from the saved sentences
    df = pd.DataFrame(st.session_state.sentences)
    # Convert DataFrame to TSV format
    tsv = df.to_csv(sep="\t", index=False)
    # Create a BytesIO object for downloading
    b = BytesIO()
    b.write(tsv.encode())
    b.seek(0)
    # Download button for the TSV file
    st.download_button(
        label="Download TSV",
        data=b,
        file_name="corrected_sentences.tsv",
        mime="text/tsv"
    )