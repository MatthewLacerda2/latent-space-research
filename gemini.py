from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import GenerateContentConfig
import logging
import os

gemini_temperature = 0.0
logger = logging.getLogger(__name__)
load_dotenv()

def get_client():    
    return Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_output(full_prompt: str) -> str:
    
    client = get_client()
    
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=full_prompt, config=GenerateContentConfig(
        response_mime_type='text/plain',
        automatic_function_calling={"disable": True}
    ),)
    
    return response.text