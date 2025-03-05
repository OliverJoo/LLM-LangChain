import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import fasttext
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import json
import pandas as pd
import logging
import textwrap
from IPython.display import display, HTML

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Model and Settings ---
# ChatOpenAI settings (using gpt-4o-mini model)
chat_openai = ChatOpenAI(model_name="gpt-4o-mini")


# --- Utility Functions ---
def detect_language(text: str) -> str:
    """Detect the language of the given text."""
    try:
        model = fasttext.load_model('./data/lid.176.ftz')
        predictions = model.predict(text, k=1)
        lang = predictions[0][0].split('__')[-1]
        return lang
    except Exception as e:
        logger.error(f"Error during language detection: {e}")
        return "unknown"


def remove_korean_suffix(text: str) -> str:
    """Remove Korean suffixes from the given text."""
    try:
        suffix_list = ["입니다.", "요.", "예요.", "입니다", "요"]
        for suffix in suffix_list:
            if text.endswith(suffix):
                return text[:-len(suffix)]
        return text
    except Exception as e:
        logger.error(f"Error removing Korean suffix: {e}")
        return text


# --- Pydantic Model ---
class SentimentAnalysis(BaseModel):
    """Pydantic model for sentiment analysis results."""
    overall_sentiment: str = Field(
        description="The overall sentiment of the review (Very Positive, Positive, Negative, Very Negative, or Neutral)")
    key_points: List[str] = Field(description="List of key points extracted from the review")


# --- Chain Definitions ---
def create_translation_chain() -> ChatOpenAI | StrOutputParser:
    """Create a chain for English to Korean translation."""
    try:
        translate_to_ko_prompt = ChatPromptTemplate.from_template(
            """You are a professional translator who is fluent in English and Korean.
            Your task is to translate the following text to Korean accurately and naturally.
            Please pay close attention to grammar and idiomatic expressions.
            You must only respond in Korean and should never add any additional English sentences to your responses.
            If the text is Korean, you must return text without any changes.
            The text to translate is: {text}"""
        )
        return translate_to_ko_prompt | chat_openai | StrOutputParser()
    except Exception as e:
        logger.error(f"Error creating English to Korean translation chain: {e}")
        raise


def create_sentiment_analysis_chain(pydantic_parser: PydanticOutputParser) -> ChatOpenAI | PydanticOutputParser:
    """Create a chain for sentiment analysis and key point extraction."""
    try:
        extract_points_sentiment_prompt = ChatPromptTemplate.from_template(
            """You are analyzing a review for the 'Book Creator Guide' GPT model. Your task is to extract key points from the given review text and determine the overall sentiment.

            Review: {text}

            Instructions:
            1. Determine the overall sentiment of the review.
            2. You must select one of the following options and reply in Korean ONLY: "매우 긍정적", "긍정적", "중립적", "부정적", or "매우 부정적".
            3. Never include any additional sentences, explanations, or examples in English.
            4. Only return the option selected.
            5. Extract up to 3 key points from the review that align with this overall sentiment.
            6. Each point must be directly derived from the review text and should reflect the tone and sentiment of the original review.
            7. If the review is very short or lacks detail, it's okay to extract fewer than 3 points.
            8. If you can't find any clear points, provide a single point stating "No specific points could be extracted from this short review."

            {format_instructions}

            Ensure that your response is a valid JSON object with 'overall_sentiment' and 'key_points' fields.

            Analysis:"""
        )
        return extract_points_sentiment_prompt | chat_openai | pydantic_parser
    except Exception as e:
        logger.error(f"Error creating sentiment analysis and key point extraction chain: {e}")
        raise


def create_translate_to_en_chain() -> ChatOpenAI | StrOutputParser:
    """Create a chain for multi-language to English translation."""
    try:
        translate_to_en_prompt = ChatPromptTemplate.from_template(
            "Translate the following text to English. If it's already in English, return it as is: {text}"
        )
        return translate_to_en_prompt | chat_openai | StrOutputParser()
    except Exception as e:
        logger.error(f"Error creating multi-language to English translation chain: {e}")
        raise


# --- Workflow ---
def create_workflow(
        translate_to_ko_chain, extract_points_sentiment_chain, translate_to_en_chain
) -> RunnablePassthrough:
    """Create the main workflow."""
    try:
        workflow = (
                # 1. Receive raw text input (input: {"text": review})
                {"text": RunnablePassthrough()}
                # 2. Detect language (output: {"lang": detected_lang, "text": original_text})
                | {"lang": lambda x: detect_language(x["text"]), "text": lambda x: x["text"]}
                # 3. Translate to English or keep original (output: {"lang": detected_lang, "text": original_text, "en_text": translated_en_text or original_text})
                | RunnablePassthrough.assign(en_text=lambda x: translate_to_en_chain.invoke({"text": x["text"]}) if x["lang"] != "en" else x["text"])
                # 4. Analyze sentiment and extract key points (output: {"lang": detected_lang, "text": original_text, "en_text": translated_en_text or original_text, "analysis": SentimentAnalysis object})
                | RunnablePassthrough.assign(analysis=lambda x: extract_points_sentiment_chain.invoke({"text": x["en_text"],"format_instructions": parser.get_format_instructions()}))
                # 5. Translate overall sentiment to Korean (output: "매우 긍정적", "긍정적", "중립적", "부정적", "매우 부정적" 중 하나)
                | {"ko_sentiment": lambda x: translate_to_ko_chain.invoke({"text": x["analysis"].overall_sentiment}),
                    # 6. Translate key points to Korean (output: [translated_ko_point1, translated_ko_point2, ...])
                    "ko_points": lambda x: [translate_to_ko_chain.invoke({"text": point}) for point in x["analysis"].key_points],# 7. Translate English review to Korean (output: translated_ko_review or original_text)
                    "ko_review": lambda x: translate_to_ko_chain.invoke({"text": x["en_text"]}) if x["lang"] != "ko" else x["text"],
                   # 8. Extract original text
                    "original_text": lambda x: x["text"],
                   # 9. Extract detected language
                    "detected_language": lambda x: x["lang"]

                }  # 10. Combine all results (output: {"ko_sentiment": translated_ko_sentiment, "ko_points": translated_ko_points, "ko_review": translated_ko_review, "original_text": original_text, "detected_language": detected_language})
                | {"원문": lambda x: x["original_text"], "감지된 언어": lambda x: x["detected_language"],
                   "한국어 리뷰": lambda x: x["ko_review"], "전체 감성": lambda x: remove_korean_suffix(x["ko_sentiment"]), "주요 포인트": lambda x: x["ko_points"]}
                # 11. Format the final results (output: {"원문": original_text, "감지된 언어": detected_language, "한국어 리뷰": translated_ko_review, "전체 감성": translated_ko_sentiment, "주요 포인트": translated_ko_points})
        )
        return workflow
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Create a Pydantic output parser
        parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)

        # Create chains
        translate_to_ko_chain = create_translation_chain()
        extract_points_sentiment_chain = create_sentiment_analysis_chain(parser)
        translate_to_en_chain = create_translate_to_en_chain()

        # Create the workflow
        workflow = create_workflow(translate_to_ko_chain, extract_points_sentiment_chain, translate_to_en_chain)

        # List of reviews
        reviews = [
            "This is FANTASTICO! I've wanted to write books my entire life, but lack the executive functioning skills to ever know where to begin. This AI book creator does all the things my ADHD brain can't and all I have to do is punch in the ideas.",
            "fluixet en la representación d'imatges",
            "Muadili diğer uygulamalar ile kıyaslanamayacak kadar güzel. Lütfen Microsoft un bu uygulamanın içine sıçmasına izin vermeyin, teşekkürler",
            "buono il risultato ma la storia dovrebbe essere maggiormente dettagliata",
            "j'adore",
            "感觉还是不行",
            "świetne",
            "no logic. no consistency. confused very easily.",
            "가톨릭에서는 마리아와 성인을 숭배하는 것이 아니라 신앙의 모범으로 공경하고 있습니다. 한국어로 숭배하다라고 해석하는 것은 신으로 숭배하는 것으로 오해를 불러일으킬 수 있는 번역입니다. 따라서 공경하다로 수정하여야 합니다.",
        ]

        # Execute the workflow
        logger.info("============ Workflow Started ============")
        results = workflow.batch([{"text": review} for review in reviews])
        logger.info("============ Workflow Completed ============")

        df = pd.DataFrame(results)
        # Print the DataFrame with adjusted column width
        pd.set_option('display.max_colwidth', 100)  # Adjust the width as needed
        pd.set_option('display.width', 1000)
        display(df)

    except Exception as e:
        logger.error(f"An unexpected error occurred during the main execution: {e}")
