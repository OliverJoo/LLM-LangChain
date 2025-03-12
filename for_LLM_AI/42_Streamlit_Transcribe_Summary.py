import os
import time
import tempfile
import streamlit as st
from pytubefix import YouTube
from openai import OpenAI, BadRequestError
import tiktoken
from pydantic import BaseModel  # Import pydantic for schema definitions
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Client Initialization ---
# Initialize OpenAI client (upstage_client removed) # 수정 : upstage_client 제거
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# --- Pydantic Model ---
class KeywordExtractionResponse(BaseModel):
    """Pydantic model for structured keyword extraction."""
    keywords: list[str]


# --- Utility Functions ---
def extract_keywords(title: str, description: str) -> str:
    """Extract keywords from the title and description of a video."""
    try:
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that extracts keywords from video titles and descriptions."},
            {"role": "user", "content": f"Title:\n{title}\n\nDescription:\n{description}\n\n"}
        ]
        response = openai_client.beta.chat.completions.parse(  # 수정 : gpt-4o-mini로 변경, openai_client로 변경
            model="gpt-4o-mini",
            messages=messages,
            response_format=KeywordExtractionResponse,  # for Structured Outputs(available models : gpt-4o & 4o-mini)
            temperature=0.5
        )
        if response.choices[0].message.parsed:
            keywords = response.choices[0].message.parsed.keywords
            return ', '.join(keywords)
        else:
            st.info("No keywords found in the content.")
            return ""
    except BadRequestError as e:
        st.error(f"An error occurred: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return ""


def get_video_info(url: str) -> tuple[str, str]:
    """Get the title and description of a YouTube video."""
    try:
        yt = YouTube(url)
        title = yt.title
        description = yt.description if yt.description else get_description_fallback(url)
        return title, description
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return "", ""


def get_description_fallback(url: str) -> str:
    """Fallback to retrieve description if not available via pytube."""
    try:
        yt = YouTube(url)
        for n in range(6):
            try:
                description = yt.initial_data["engagementPanels"][n]["engagementPanelSectionListRenderer"]["content"][
                    "structuredDescriptionContentRenderer"]["items"][1]["expandableVideoDescriptionBodyRenderer"][
                    "attributedDescriptionBodyText"]["content"]
                return description
            except KeyError:
                continue
        return ""
    except Exception as e:
        logger.error(f"Error getting fallback description: {e}")
        return ""


def download_audio(url: str) -> str:
    """Download only audio stream from a YouTube video."""
    logger.info("Downloading audio...")
    max_retries = 3
    retry_delay = 5  # seconds
    for retry in range(max_retries):
        try:
            yt = YouTube(url)
            audio_stream = yt.streams.get_audio_only()
            if not audio_stream:
                raise Exception("No audio streams available for this video.")
            output_file = audio_stream.download()
            return output_file
        except Exception as e:
            logger.error(f"Error occurred while downloading audio: {str(e)}")
            if retry < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds... (Attempt {retry + 1} of {max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Max retries reached. Unable to download audio.")
                st.error(f"Failed to download audio: {e}")
                raise
    return ""


def trim_file_to_size(filepath: str, max_size: int) -> str:
    """Trim a file to a specified maximum size."""
    try:
        file_size = os.path.getsize(filepath)
        logger.info(f"File size: {file_size} bytes")
        if file_size <= max_size:
            return filepath
        logger.info(f"File size exceeds the maximum size of {max_size} bytes. Trimming the file...")
        _, file_ext = os.path.splitext(filepath)
        with open(filepath, "rb") as file:
            data = file.read(max_size)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(data)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"Error trimming file: {e}")
        return ""


def transcribe(audio_filepath: str, language: str = None, response_format: str = 'text', prompt: str = None):
    """Transcribe an audio file using OpenAI's Whisper API."""
    try:
        MAX_FILE_SIZE = 26_210_000  # whisper-1 model size limit : 25MB
        trimmed_audio_filepath = trim_file_to_size(audio_filepath, MAX_FILE_SIZE)
        logger.info("Transcribing audio...")
        with open(trimmed_audio_filepath, "rb") as file:
            kwargs = {'file': file, 'model': "whisper-1", 'response_format': response_format}
            if language is not None:
                kwargs['language'] = language
            if prompt is not None:
                kwargs['prompt'] = prompt
            transcript = openai_client.audio.transcriptions.create(**kwargs)  # 수정 : upstage_client 제거
        st.session_state.transcript = transcript
        if trimmed_audio_filepath != audio_filepath:
            os.remove(trimmed_audio_filepath)
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        st.error(f"Failed to transcribing audio: {e}")


def get_tokenizer(model_name: str):
    """Get the tokenizer for a given model name."""
    if model_name.startswith("gpt"):
        return tiktoken.encoding_for_model(model_name)
    else:
        return tiktoken.encoding_for_model("gpt-4o-mini")


def split_into_chunks(text: str, max_tokens: int, tokenizer) -> list[str]:
    """
        * Split text into chunks based on token limit.
        * Lengthy text can cause hallucinations. Thus, limiting text length is important.
    """
    logger.info(f"Checking if transcript fits within the token limit of {max_tokens}...")
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    for paragraph in paragraphs:
        paragraph_tokens = len(tokenizer.encode(paragraph))
        if current_chunk_tokens + paragraph_tokens <= max_tokens:
            current_chunk.append(paragraph)
            current_chunk_tokens += paragraph_tokens
        else:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_tokens = paragraph_tokens
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    if len(chunks) > 1:
        logger.info(f"Split transcript into {len(chunks)} chunks.")
    return chunks


def translate(source_text: str, source_language_code: str, target_language_name: str) -> str:
    """Translate text using OpenAI's ChatCompletion API."""
    try:
        logger.info(f"Translating {source_language_code} text into {target_language_name} ...")
        content_type = "Translation"
        temperature = 0.4
        preferred_model = "gpt-4o-mini"  # 수정 : 항상 gpt-4o-mini를 활용하도록 변경하였습니다.
        fallback_model = "gpt-4o-mini"  # 수정 : 항상 gpt-4o-mini를 활용하도록 변경하였습니다.
        logger.info(f"Preferred model: {preferred_model}")
        logger.info(f"Fallback model: {fallback_model}")
        client = openai_client  # 수정 : upstage_client 제거
        chunk_size = 1024
        tokenizer = get_tokenizer("gpt-4o-mini")
        chunks = split_into_chunks(source_text, chunk_size, tokenizer)
        results = []
        progress_text = "Translation progress"
        progress_bar = None
        if len(chunks) > 1:
            progress_bar = st.progress(0, text=progress_text)  # show progress bar
        for i, chunk in enumerate(chunks, start=1):
            messages = [
                {
                    "role": "system",
                    "content": f"You are a professional translator who is fluent in {source_language_code} and {target_language_name}. Your task is to translate the following text to {target_language_name} accurately and naturally. Please pay close attention to grammar and idiomatic expressions. You must only respond in {target_language_name} and should never add any additional sentences to your responses. If the text is already in {target_language_name}, you must return text without any changes. The text to translate is:"
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ]
            progress = i / len(chunks)
            try:
                logger.info(f"Attempting to translate chunk {i} using {preferred_model} ...")
                response = client.chat.completions.create(model=preferred_model, messages=messages,
                                                          temperature=temperature)
            except BadRequestError as e:
                logger.error(f"BadRequestError occurred: {e}")
                logger.info(f"Retrying translation chunk {i} using {fallback_model} ...")
                response = client.chat.completions.create(model=fallback_model, messages=messages,
                                                          temperature=temperature)
            finally:
                translated_chunk = response.choices[0].message.content
                results.append(translated_chunk)
                if progress_bar:
                    progress_bar.progress(progress, text=f"{progress_text} {progress:.0%}")
        logger.info("Translation completed.\n")
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        st.error(f"Failed to transalte: {e}")
        return ""


def generate_content(content_type: str, content_language: str, transcript_language_code: str, transcript_format: str,
                     transcript: str) -> str:
    """Generate content based on the selected type and the transcript."""
    try:
        if content_type == "Translation":
            return translate(transcript, transcript_language_code, content_language)
        preferred_model = "gpt-4o-mini"  # 수정 : gpt-4o-mini를 활용하도록 변경하였습니다.
        fallback_model = "gpt-4o-mini"  # 수정 : gpt-4o-mini를 활용하도록 변경하였습니다.
        temperature = 0.5
        if content_type in ["Essay", "Blog article", "Comment on Key Moments"]:
            temperature = 0.8
        elif content_type in ["Simple Summary", "Detailed Summary"]:
            temperature = 0.4
        if transcript_format == "srt":
            transcript = extract_dialogues_from_srt(transcript)
        logger.info(f"Generating {content_type} in {content_language} with temperature {temperature} ...")
        messages = [
            {"role": "system", "content": prompt[content_type].format(language=content_language)},
            {"role": "user", "content": (f"Transcript:\n{transcript}\n\n" f"{content_type} in {content_language}:")}
        ]
        result = ""
        try:
            logger.info(f"Attempting to generate content using {preferred_model} ...")
            client = openai_client  # 수정 : upstage_client 제거
            response = client.chat.completions.create(model=preferred_model, messages=messages, temperature=temperature)
            result = response.choices[0].message.content
        except BadRequestError as e:
            logger.error(f"BadRequestError occurred: {e}")
            logger.info(f"Retrying content generation using {fallback_model} ...")
            client = openai_client  # 수정 : upstage_client 제거
            response = client.chat.completions.create(model=fallback_model, messages=messages, temperature=temperature)
            result = response.choices[0].message.content
        logger.info("Content generated.\n")
        return result
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        st.error(f"Failed to generate content: {e}")
        return ""


def extract_dialogues_from_srt(srt_content: str) -> str:
    """Extract dialogue lines from an SRT content string."""
    try:
        """
        lines = [
            "1",  # 자막 인덱스
            "00:00:00,000 --> 00:00:02,500",  # 타임스탬프
            "Hello, world!",  # 대화 내용
            "",  # 빈 줄
            "2",  # 자막 인덱스
            "00:00:03,000 --> 00:00:05,000",  # 타임스탬프
            "How are you?"  # 대화 내용
        ]
        """
        lines = srt_content.strip().split('\n')
        dialogues = [lines[i] for i in range(2, len(lines), 4)]
        return '\n'.join(dialogues)
    except Exception as e:
        logger.error(f"Error extracting dialogues from SRT content: {e}")
        return ""


# --- Streamlit UI ---
st.title("YouTube Transcription and Content Generation")

url = st.text_input("Enter Video URL:")

# Load Video Info
if st.button("Load Video Info"):
    if url:
        title, description = get_video_info(url)
        transcription_prompt = extract_keywords(title, description)
        st.session_state.video_info = transcription_prompt
    else:
        st.error("Please enter a valid YouTube URL.")

prompt = st.text_area(
    "What's in the video? (Optional)",
    value=st.session_state.get('video_info', ''),
    help=(
        "Provide a brief description of the video or include specific terms like "
        "unique names and key topics to enhance accuracy. This can include spelling "
        "out hard-to-distinguish proper nouns."
    )
)

language_mapping = {'한국어': 'ko', 'English': 'en', '日本語': 'ja', '中文': 'zh', 'Español': 'es', 'Français': 'fr',
                    'Deutsch': 'de'}
video_language_name = st.selectbox("Select Language of the Video (optional):",
                                   ['', '한국어', 'English', '日本語', '中文', 'Español', 'Français', 'Deutsch'])
st.session_state.video_language = language_mapping.get(video_language_name, '')
st.session_state.response_format = st.selectbox("Select Output Format:", ('text', 'srt', 'vtt'))

if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""

# Transcribe Video
if st.button("Transcribe Video"):
    if url:
        with st.spinner('Downloading and transcribing video... This may take a while.'):
            filename = download_audio(url)
            transcribe(filename, language=st.session_state.video_language,
                       response_format=st.session_state.response_format, prompt=prompt)
            os.remove(filename)
        st.success('Done! Subtitles have been generated.')
        logger.info("Transcription completed.")
    else:
        st.error("Please enter a URL.")

# Display subtitles if available
if st.session_state.transcript:
    st.text_area("Subtitles:", value=st.session_state.transcript, height=300)
