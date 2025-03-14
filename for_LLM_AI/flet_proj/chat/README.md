# A chat Flet app

An example of a minimal Flet app.

To run the app:

```
flet run [app_directory]
```

# Flet Chat Application Examples

This folder contains example code for simple chat applications built using the Flet library. Each
file (`chat1.py`, `chat2.py`, `chat3.py`, `chat4.py`) demonstrates a progressively enhanced way of implementing a chat
application.

## File Descriptions

### chat1.py

- **Functionality:** Implements the most basic form of a chat application.
- **Features:**
    - Provides a field for users to enter their name.
    - Includes a field for entering and sending messages, along with a send button.
    - Uses the `pubsub` feature to send and receive messages.
    - Displays all messages as plain text.
    - Implements the most fundamental chat functionality.
- **How to Run:** Execute by typing `python chat1.py`.

### chat2.py

- **Functionality:** Enhances `chat1.py` by adding a dialog box for users to enter their name before joining the chat.
- **Features:**
    - Displays a dialog box for entering the user's name when the application starts.
    - Shows an error message if the user tries to join without entering a name.
    - Stores the user's entered name in the session, which is then used when sending messages.
    - Uses `page.session` to manage the user's name.
    - Includes basic error handling mechanisms.
- **How to Run:** Execute by typing `python chat2.py`.

### chat3.py

- **Functionality:** Extends the functionality of `chat2.py` by introducing message objects and handling login messages
  separately.
- **Features:**
    - Introduces the `Message` class to manage messages as objects.
    - Adds a `login` message type, which is handled specially when a user joins the chat.
    - Displays `login` messages in blue, italic text.
    - Utilizes the `Message` class to generate message objects.
    - Utilizes the `to_dict()` and `from_dict()` methods to convert messages.
- **How to Run:** Execute by typing `python chat3.py`.

### chat4.py

- **Functionality:** Improves on `chat3.py` by adding the `ChatMessage` class to display avatars and messages together.
- **Features:**
    - Introduces the `ChatMessage` class to display user avatars alongside messages.
    - Each user has a unique avatar color.
    - Avatars display the user's initials.
    - Utilizes the `ChatMessage` class to manage messages as objects.
    - Assigns avatar colors using the user's name.
    - Implements avatars using the `get_initials()` and `get_avatar_color()` functions.
- **How to Run:** Execute by typing `python chat4.py`.

## Common Execution Instructions

These applications are built on top of the Flet library. Therefore, you must have the Flet library installed before
running them.