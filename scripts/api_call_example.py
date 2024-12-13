from openai import OpenAI

client = OpenAI(api_key="0",base_url="http://192.168.0.13:8000/v1")
messages = [{"role": "user", "content": "Who are you?"}]
result = client.chat.completions.create(messages=messages,model="../Models/Llama-3.2-1B-Instruct")
print(result.choices[0].message.content)