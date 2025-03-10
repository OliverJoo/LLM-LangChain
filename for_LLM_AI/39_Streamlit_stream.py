import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')
# stream = True : Streaming answer, available in real-time before completion continually
response = model.generate_content("인공지능에 대해 자세히 설명해줘.", stream=True)

for no, chunk in enumerate(response, start=1):
    print(f"{no}: {chunk.text}")
    print("=" * 50)
