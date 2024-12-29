!pip install python-dotenv requests groq
import os, requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # take environment variables from .env.
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set.")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Why is the weather in Ottawa colder than Tampa?",
        }
    ],
    model="llama3-8b-8192",
)
print(chat_completion.choices[0].message.content)