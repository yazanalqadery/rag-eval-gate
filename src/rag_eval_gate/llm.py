"""All OpenRouter API calls go through this module: one client, one place for model names."""

import os

from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"

def get_client() -> OpenAI:
    """Create the client only when needed, so importing this module never needs an API key."""
    load_dotenv()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],  # KeyError with the name if missing: fail loudly
    )

def embed(texts: list[str]) -> list[list[float]]:
    """One vector per text, in ONE API call."""
    response = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]

def chat(system: str, user: str, temperature: float = 0) -> str:
    """Send a system + user message, return the reply text."""
    response = get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return response.choices[0].message.content