import openai
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# text-embedding-3-smallをデフォルトで利用
def get_embeddings(sentences: List[str], model: str = "text-embedding-3-small") -> List[list]:
    response = openai.embeddings.create(
        input=sentences,
        model=model
    )
    return [d.embedding for d in response.data] 