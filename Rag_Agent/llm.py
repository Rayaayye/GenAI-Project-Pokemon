from langchain_google_genai import ChatGoogleGenerativeAI
import os


# Initializes and returns the Google Gemini language model instance
def get_llm():
    # Create a ChatGoogleGenerativeAI instance
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # Use the Gemini 2.5 Flash Lite model
        temperature=0.2,  # We put low temperature to get more deterministic and factual responses
        max_output_tokens=750,  # Limit output length to 750 tokens
    )
    # Return the configured LLM instance
    return llm


    

