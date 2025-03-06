from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes

# Initialization FastAPI App
app = FastAPI()

openai_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Creation Prompt Template
prompt = ChatPromptTemplate.from_template("{topic}에 관해 노랫말을 써줘.")

# Chaining Prompt and Model
chain = prompt | openai_model

# add routes
add_routes(app, chain, path="/lyrics")

# test url : http://localhost:8000/lyrics/playground/
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

