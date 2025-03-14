import flet as ft


class Message:
    """
    Represents a chat message with user, text, and message type.
    """

    def __init__(self, user, text, message_type):
        """
        Initializes a Message object.

        Args:
            user (str): The name of the user.
            text (str): The text content of the message.
            message_type (str): The type of the message (e.g., "login", "chat").
        """
        self.user = user
        self.text = text
        self.message_type = message_type

    def to_dict(self):
        """
        Converts the Message object to a dictionary.

        Returns:
            dict: A dictionary representation of the Message.
        """
        return {"user": self.user, "text": self.text, "message_type": self.message_type}

    @staticmethod
    def from_dict(data):
        """
        Creates a Message object from a dictionary.

        Args:
            data (dict): A dictionary containing the message data.

        Returns:
            Message: A Message object.
        """
        return Message(data["user"], data["text"], data["message_type"])


class ChatMessage(ft.Row):
    """
    Displays a chat message with the user's avatar and message text.
    """

    def __init__(self, user_name, display_text):
        """
        Initializes a ChatMessage object.

        Args:
            user_name (str): The name of the user.
            display_text (str): The text content of the message.
        """
        super().__init__()
        self.vertical_alignment = ft.CrossAxisAlignment.START  # Align the row to the start vertically
        self.controls = [
            ft.CircleAvatar(
                content=ft.Text(self.get_initials(user_name)),  # Display user's initials in a circle
                color=ft.colors.WHITE,  # Set text color to white
                bgcolor=self.get_avatar_color(user_name),  # Set background color based on user name
            ),
            ft.Column(
                [
                    ft.Text(user_name, weight=ft.FontWeight.BOLD),  # Display user name in bold
                    ft.Text(display_text, selectable=True),  # Display message text, selectable for copying
                ],
                tight=True,  # Remove extra spacing between text controls
                spacing=5,  # Set spacing between user name and message text
            ),
        ]

    def get_initials(self, user_name):
        """
        Gets the initials of the user name.

        Args:
            user_name (str): The full user name.

        Returns:
            str: The initials of the user name.
        """
        return user_name[:1].upper()  # Return the first letter in uppercase

    def get_avatar_color(self, user_name):
        """
        Determines the avatar color based on the user name.

        Args:
            user_name (str): The full user name.

        Returns:
            str: The avatar color.
        """
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
        return colors_lookup[hash(user_name) % len(colors_lookup)]  # Use hash to consistently assign colors


def main(page: ft.Page):
    """
    Main function to set up the chat application UI and handle user interactions.

    Args:
        page (ft.Page): The Flet page object.
    """
    page.title = "Flet Chat"  # Set the title of the application window

    # Create a text field for entering the user's name with autofocus
    user_name_field = ft.TextField(label="Enter your name", autofocus=True)

    def join_click(e):
        """
        Handles the join button click event.

        Checks if the user has entered a name. If not, displays an error.
        Otherwise, sets the user's name in the session, closes the dialog,
        sends a login message, and updates the UI.

        Args:
            e: The event data.
        """
        if not user_name_field.value:
            user_name_field.error_text = "Please enter your name!"  # Display error message if no name is entered
            user_name_field.update()  # Update the UI to show the error
        else:
            page.session.set("user_name", user_name_field.value)  # Store the user's name in the session
            dialog.open = False  # Close the dialog

            # Send login message
            login_message = Message(
                user=user_name_field.value,
                text=f"{user_name_field.value} has joined the chat.",
                message_type="login",
            )
            page.pubsub.send_all(login_message.to_dict())  # Send login message to all subscribers
            page.update()  # Update the UI

    # Create a dialog for user name input
    dialog = ft.AlertDialog(
        modal=True,  # Make the dialog modal (blocks interaction with the rest of the UI)
        title=ft.Text("Welcome!"),  # Set the dialog title
        content=ft.Column([user_name_field], tight=True),  # Add the user name input field to the dialog
        actions=[ft.ElevatedButton(text="Join", on_click=join_click)],  # Add a "Join" button
        actions_alignment="end",  # Align the actions to the end of the dialog
    )
    page.overlay.append(dialog)  # Add the dialog to the page overlay
    dialog.open = True  # Open the dialog

    # Message Handling Function
    def on_message(msg_dict):
        """
        Callback function to handle incoming messages.

        Adds received messages to the messages column and updates the UI.
        Handles both "login" and "chat" message types.

        Args:
            msg_dict (dict): A dictionary containing the message data.
        """
        msg = Message.from_dict(msg_dict)  # Create a Message object from the dictionary
        if msg.message_type == "login":  # Check if it's a login message
            messages.controls.append(
                ft.Text(msg.text, italic=True, color=ft.colors.BLUE)
            )  # Add login message with specific style
        elif msg.message_type == "chat":  # Check if it's a chat message
            messages.controls.append(ChatMessage(msg.user, msg.text))  # Add chat message with user name
        page.update()  # Update the UI

    # Subscribe to the PubSub to receive messages
    page.pubsub.subscribe(on_message)

    def send_click(e):
        """
        Handles the send button click event.

        Retrieves the user's name from the session, sends the message with the user's name,
        clears the input field, and updates the UI. If no user name is set, opens the dialog.

        Args:
            e: The event data.
        """
        user_name = page.session.get("user_name")  # Retrieve the user's name from the session
        if user_name:
            chat_message = Message(
                user=user_name, text=message_field.value, message_type="chat"
            )
            page.pubsub.send_all(chat_message.to_dict())  # Send the message to all subscribers with user name
            message_field.value = ""  # Clear the input field after sending
            page.update()  # Update the UI to clear the input field
        else:
            dialog.open = True  # Open the dialog if no user name is set
            page.update()  # Update the UI

    messages = ft.Column()  # Create a column to hold the chat messages
    message_field = ft.TextField(hint_text="Enter your message...",
                                 expand=True)  # Create a text field for entering messages
    send = ft.ElevatedButton("Send", on_click=send_click)  # Create a send button

    # Add the components to the page layout
    page.add(messages, ft.Row(controls=[message_field, send]))


ft.app(target=main)
