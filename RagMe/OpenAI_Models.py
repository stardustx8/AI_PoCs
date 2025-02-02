import os
import httpx
from openai import OpenAI

def create_client(api_key: str) -> OpenAI:
    """
    Instantiates and returns a new OpenAI client using the provided API key.
    """
    return OpenAI(api_key=api_key)

# Option 1: Use an environment variable
api_key = os.environ.get("OPENAI_API_KEY", "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")
client = create_client(api_key)

# Make a GET request to the /models endpoint, casting the response to an httpx.Response.
response = client.get("/models", cast_to=httpx.Response)

# Print the JSON content from the response.
models = response.json()
print(models)
