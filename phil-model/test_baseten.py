import requests
import os

# Replace the empty string with your model id below
model_id = "dq4xkkdw"
baseten_api_key = os.environ["BASETEN_API_KEY"]

data = {
    "messages": [
        {"role": "system", "content": "You are a knowledgable, engaging, geology teacher."},
        {"role": "user", "content": "What is the impact of the Mistral wind on the French climate?"},
    ],
    "stream": True,
    "max_new_tokens": 512,
    "temperature": 0.9
}

# Call model endpoint
res = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json=data,
    stream=True
)

# Print the generated tokens as they get streamed
for content in res.iter_content():
    print(content.decode("utf-8"), end="", flush=True)