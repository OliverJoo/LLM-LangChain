import os

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import streamlit as st

# Set the title of the Streamlit app
st.title("Streamlit LangChain Chatbot Example")

# Initialize StreamlitChatMessageHistory for managing the chat history
msgs = StreamlitChatMessageHistory()
# Add an initial AI message if the chat history is empty
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello, how can I help you?")

# Configure the LangChain prompt
# - A system message that sets the behavior of the chatbot.
# - A placeholder for the chat history.
# - A user message template that receives the user's question.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Initialize the ChatOpenAI model with the 'gpt-4o-mini' model and the OpenAI API key
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv('OPENAI_API_KEY')
)

# Create the LangChain chain by combining the prompt and the language model
chain = prompt | llm
# Wrap the chain with RunnableWithMessageHistory to manage chat history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Function to retrieve chat history
    input_messages_key="question",  # Key for user's question
    history_messages_key="history",  # Key for the chat history
)

# Render the current messages in the StreamlitChatMessageHistory
# Display each message in the chat history in the Streamlit UI
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Handle new user input prompts
if prompt := st.chat_input():
    # Display the user's prompt in the chat UI
    st.chat_message("human").write(prompt)
    # New messages are automatically recorded by RunnableWithMessageHistory
    # Configure the session ID for the chat history (in this case, "any")
    config = {"configurable": {"session_id": "any"}}
    # Invoke the chain with the user's prompt and the session configuration
    response = chain_with_history.invoke({"question": prompt}, config)
    # Display the AI's response in the chat UI
    st.chat_message("ai").write(response.content)