from langchain_google_genai import ChatGoogleGenerativeAI
import os


def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        max_output_tokens=750,
    )
    return llm


    

