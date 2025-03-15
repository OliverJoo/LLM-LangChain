import flet as ft
import os
from openai import OpenAI, BadRequestError
from pydantic import BaseModel
import logging

# --- Logging Configuration ---
# Configure basic logging settings to monitor the application's behavior.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- OpenAI Client Initialization ---
# Initialize the OpenAI client with API key from environment variables.
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    raise ValueError("Failed to initialize OpenAI client. Please check your API key.") from e


# --- Message Class Definition ---
class Message:
    """Represents a chat message with user, text, message type, and translations."""

    def __init__(self, user, text, message_type, translations=None):
        self.user = user
        self.text = text
        self.message_type = message_type
        self.translations = translations or {}

    def to_dict(self):
        """Converts the Message object to a dictionary."""
        return {
            "user": self.user,
            "text": self.text,
            "message_type": self.message_type,
            "translations": self.translations,
        }

    @staticmethod
    def from_dict(data):
        """Creates a Message object from a dictionary."""
        return Message(
            data["user"],
            data["text"],
            data["message_type"],
            data.get("translations", {}),
        )


# --- Chat Message Display Classes ---
class ChatMessage(ft.Row):
    """Displays a chat message from the current user."""

    def __init__(self, user_name, display_text):
        super().__init__()
        self.vertical_alignment = ft.CrossAxisAlignment.START
        self.controls = [
            ft.CircleAvatar(
                content=ft.Text(self.get_initials(user_name)),
                color=ft.colors.WHITE,
                bgcolor=self.get_avatar_color(user_name),
            ),
            ft.Column(
                [
                    ft.Text(user_name, weight=ft.FontWeight.BOLD),
                    ft.Text(display_text, selectable=True),
                ],
                tight=True,
                spacing=5,
            ),
        ]

    def get_initials(self, user_name):
        """Gets the initials of the username."""
        return user_name[:1].upper()

    def get_avatar_color(self, user_name):
        """Determines the avatar color based on the username."""
        colors_lookup = [
            ft.colors.AMBER,
            ft.colors.BLUE,
            ft.colors.BROWN,
            ft.colors.CYAN,
            ft.colors.GREEN,
            ft.colors.INDIGO,
            ft.colors.LIME,
            ft.colors.ORANGE,
            ft.colors.PINK,
            ft.colors.PURPLE,
            ft.colors.RED,
            ft.colors.TEAL,
            ft.colors.YELLOW,
        ]
        return colors_lookup[hash(user_name) % len(colors_lookup)]


class ChatMessageWithTranslation(ft.Row):
    """Displays a chat message from other users with original and translated text."""

    def __init__(self, user_name, original_text, translated_text):
        super().__init__()
        self.vertical_alignment = ft.CrossAxisAlignment.START
        self.controls = [
            ft.CircleAvatar(
                content=ft.Text(self.get_initials(user_name)),
                color=ft.colors.WHITE,
                bgcolor=self.get_avatar_color(user_name),
            ),
            ft.Column(
                [
                    ft.Text(user_name, weight=ft.FontWeight.BOLD),
                    ft.Text(f"Original: {original_text}", selectable=True),
                    ft.Text(f"Translated: {translated_text}", selectable=True),
                ],
                tight=True,
                spacing=5,
            ),
        ]

    def get_initials(self, user_name):
        """Gets the initials of the username."""
        return user_name[:1].upper()

    def get_avatar_color(self, user_name):
        """Determines the avatar color based on the username."""
        colors_lookup = [
            ft.colors.AMBER,
            ft.colors.BLUE,
            ft.colors.BROWN,
            ft.colors.CYAN,
            ft.colors.GREEN,
            ft.colors.INDIGO,
            ft.colors.LIME,
            ft.colors.ORANGE,
            ft.colors.PINK,
            ft.colors.PURPLE,
            ft.colors.RED,
            ft.colors.TEAL,
            ft.colors.YELLOW,
        ]
        return colors_lookup[hash(user_name) % len(colors_lookup)]


# --- Pydantic Model for Translation ---
class TranslationResponse(BaseModel):
    """Pydantic model for structured translation responses."""

    translation: str


# --- Main Function ---
def main(page: ft.Page):
    """Main function to run the Flet application."""

    # Set page title
    page.title = "Multilingual Chat"

    # Initialize conversation history if not present
    if not hasattr(page.session, "conversation_history"):
        page.session.set("conversation_history", [])

    # --- UI Elements ---
    user_name_field = ft.TextField(label="Enter your name", autofocus=True)
    user_language_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("en", "English"),
            ft.dropdown.Option("ko", "한국어"),
            ft.dropdown.Option("ja", "日本語"),
        ],
        label="Select your language",
    )

    # --- Event Handlers ---
    def join_click(e):
        """Handles the join chat button click."""
        if not user_name_field.value:
            user_name_field.error_text = "Name cannot be blank!"
            user_name_field.update()
        elif not user_language_dropdown.value:
            user_language_dropdown.error_text = "Please select a language!"
            user_language_dropdown.update()
        else:
            page.session.set("user_name", user_name_field.value)
            page.session.set("user_language", user_language_dropdown.value)
            dialog.open = False
            # Send login message
            login_message = Message(
                user=user_name_field.value,
                text=f"{user_name_field.value} has joined the chat.",
                message_type="login"
            )
            page.pubsub.send_all(login_message.to_dict())
            page.update()

    dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Welcome!"),
        content=ft.Column([user_name_field, user_language_dropdown], tight=True),
        actions=[ft.ElevatedButton(text="Join chat", on_click=join_click)],
        actions_alignment="end",
    )
    page.overlay.append(dialog)
    dialog.open = True

    def on_message(msg_dict):
        """Handles incoming messages."""
        msg = Message.from_dict(msg_dict)
        current_user = page.session.get("user_name")
        user_language = page.session.get("user_language")

        # Update conversation history
        conversation_history = page.session.get("conversation_history")
        conversation_history.append(msg)
        page.session.set("conversation_history", conversation_history)

        if msg.message_type == "login":
            messages.controls.append(
                ft.Text(msg.text, italic=True, color=ft.colors.BLUE)
            )
        elif msg.message_type == "chat":
            if msg.user == current_user:
                # Display current user's message
                messages.controls.append(ChatMessage(msg.user, msg.text))
            else:
                # Display other user's message with translations
                original_text = msg.text
                translated_text = msg.translations.get(user_language, msg.text)
                messages.controls.append(
                    ChatMessageWithTranslation(msg.user, original_text, translated_text))
        page.update()

    page.pubsub.subscribe(on_message)

    def translate_text(text, target_language, conversation_history):
        """Translates text to the target language using OpenAI API."""
        logger.info(f"Starting to translate text to {target_language}...")
        # Construct messages for translation, including conversation context
        messages_for_translation = [
            {
                "role": "system",
                "content": (
                    f"Translate the following message into {target_language}. "
                    "Consider the context of the conversation and adapt the message to be easily understood and culturally appropriate for the listener. Ensure that idioms and expressions are translated in a way that makes sense in the listener's culture."
                )
            }
        ]

        # Include recent conversation history. Only include up to 5 messages
        N = 5
        recent_history = conversation_history[-N:]
        conversation_str = ''
        for msg in recent_history:
            sender = 'system' if msg.message_type == 'login' else msg.user
            content = msg.text
            conversation_str += f"{{'{sender}': '{content}'}}, "  # removed "\n" to avoid empty lines

        # Prepare the final prompt
        final_prompt = (
            f"Here is the conversation history:\n\n{conversation_str.strip(', ')}\n"
            f"Please translate the last message to {target_language}, considering the conversation context and adapting it to be easily understood and culturally appropriate for the listener:\n\n{text}"
        )

        # Remove the extra '\n'
        final_prompt = final_prompt.replace('\n\n\n', '\n\n')

        messages_for_translation.append(
            {
                "role": "user",
                "content": final_prompt
            }
        )

        logger.info(f"messages_for_translation : {messages_for_translation}")
        try:
            # Request translation from OpenAI API
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages_for_translation,
                response_format=TranslationResponse,
            )

            translation_response = completion.choices[0].message.parsed
            logger.info(
                f"Translation completed successfully. translated text : {translation_response.translation.strip()}")
            return translation_response.translation.strip()
        except BadRequestError as e:
            logger.error(f"BadRequestError occurred during translation: {e}")
            return ""
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return ""

    def send_click(e):
        """Handles sending a new chat message."""
        user_name = page.session.get("user_name")
        user_language = page.session.get("user_language")
        if user_name:
            original_text = message_field.value.strip()
            if original_text:
                # Get conversation history and temporarily add the current message
                conversation_history = page.session.get("conversation_history")
                temp_conversation_history = conversation_history + [Message(
                    user=user_name,
                    text=original_text,
                    message_type="chat",
                )]

                target_languages = ["en", "ko", "ja"]
                translations = {}
                for lang in target_languages:
                    if lang != user_language:
                        try:
                            translated_text = translate_text(
                                original_text, lang, temp_conversation_history
                            )
                            translations[lang] = translated_text
                        except Exception as ex:
                            logger.error(f"Translation error: {ex}")
                            translations[lang] = ''
                    else:
                        # No need to translate to the user's own language
                        continue

                # Create and send the message
                chat_message = Message(
                    user=user_name,
                    text=original_text,
                    message_type="chat",
                    translations=translations,
                )
                page.pubsub.send_all(chat_message.to_dict())
                message_field.value = ""
                message_field.update()
                page.update()
        else:
            dialog.open = True
            page.update()

    # --- Layout ---
    messages = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
    )
    message_field = ft.TextField(
        hint_text="Write a message...",
        autofocus=True,
        shift_enter=True,
        min_lines=1,
        max_lines=5,
        filled=True,
        expand=True,
        on_submit=send_click,
    )
    send_button = ft.IconButton(
        icon=ft.icons.SEND_ROUNDED,
        tooltip="Send message",
        on_click=send_click,
    )

    page.add(
        ft.Container(
            content=messages,
            border=ft.border.all(1, ft.colors.OUTLINE),
            border_radius=5,
            padding=10,
            expand=True,
        ),
        ft.Row(
            [message_field, send_button],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
    )


ft.app(target=main, view=ft.AppView.WEB_BROWSER)
