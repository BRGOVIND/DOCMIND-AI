import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def ask_llm(context: str, question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=api_key)

    prompt = f"""
You are DocMind AI, a careful document question-answering assistant.
Use only the provided document context. If the context does not contain
enough information, say that clearly instead of guessing.

Context:
{context}

Question:
{question}

Answer in a clear, structured way. Mention document names or page numbers
when they help the user verify the answer.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using retrieved PDF context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
