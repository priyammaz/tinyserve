"""
Once the model is served with 

    python serve.py serve --port 8000

we can test inference!
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",
)

stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Describe the transformer model"}],
    max_tokens=1024,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content is not None:
        print(delta.content, end="", flush=True)

print()