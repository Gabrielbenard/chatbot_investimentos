from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

model_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.environ.get("GEMINI_API_KEY")
)

model_qwen = ChatGroq(
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
    model="qwen/qwen3-32b"
)