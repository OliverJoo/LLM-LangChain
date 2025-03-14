import flet as ft

def main(page: ft.Page):
    """
    Main function to set up the chat application UI and handle user interactions.

    Args:
        page (ft.Page): The Flet page object.
    """
    page.title = "Flet Chat"  # Set the title of the application window

    def on_message(msg):
        """
        Callback function to handle incoming messages.

        Args:
            msg (str): The message content received.
        """
        messages.controls.append(ft.Text(msg))  # Add the received message to the messages column
        page.update()  # Update the UI to display the new message

    # Subscribe to the PubSub to receive messages
    page.pubsub.subscribe(on_message)

    def send_click(e):
        """
        Callback function to handle the 'send' button click event.

        Args:
            e: The event data.
        """
        page.pubsub.send_all(f"{user.value}: {message_field.value}")  # Send the message to all subscribers with user name
        message_field.value = ""  # Clear the input field after sending
        page.update()  # Update the UI to clear the input field

    messages = ft.Column()  # Create a column to hold the chat messages
    user = ft.TextField(hint_text="Name", width=150)  # Create a text field for the user to enter their name
    message_field = ft.TextField(hint_text="Enter your message...", expand=True)  # Create a text field for entering messages
    send = ft.ElevatedButton("Send", on_click=send_click)  # Create a send button

    # Add the components to the page layout
    page.add(messages, ft.Row(controls=[user, message_field, send]))

ft.app(target=main)