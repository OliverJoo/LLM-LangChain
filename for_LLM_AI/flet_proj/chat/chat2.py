import flet as ft

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
        Otherwise, sets the user's name in the session, closes the dialog, and updates the UI.

        Args:
            e: The event data.
        """
        if not user_name_field.value:
            user_name_field.error_text = "Please enter your name!"  # Display error message if no name is entered
            user_name_field.update()  # Update the UI to show the error
        else:
            page.session.set("user_name", user_name_field.value)  # Store the user's name in the session
            dialog.open = False  # Close the dialog
            page.update()  # Update the UI to reflect the changes

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

    def on_message(msg):
        """
        Callback function to handle incoming messages.

        Adds received messages to the messages column and updates the UI.

        Args:
            msg (str): The message content received.
        """
        messages.controls.append(ft.Text(msg))  # Add the message to the messages column
        page.update()  # Update the UI to display the new message

    # Subscribe to the PubSub to receive messages
    page.pubsub.subscribe(on_message)

    def send_click(e):
        """
        Handles the send button click event.

        Retrieves the user's name from the session, sends the message with the user's name,
        clears the input field, and updates the UI.

        Args:
            e: The event data.
        """
        user_name = page.session.get("user_name")  # Retrieve the user's name from the session
        page.pubsub.send_all(f"{user_name}: {message_field.value}")  # Send the message to all subscribers with user name
        message_field.value = ""  # Clear the input field after sending
        page.update()  # Update the UI to clear the input field

    messages = ft.Column()  # Create a column to hold the chat messages
    message_field = ft.TextField(hint_text="Enter your message...", expand=True)  # Create a text field for entering messages
    send = ft.ElevatedButton("Send", on_click=send_click)  # Create a send button

    # Add the components to the page layout
    page.add(messages, ft.Row(controls=[message_field, send]))

ft.app(target=main)