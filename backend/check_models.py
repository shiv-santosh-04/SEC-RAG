import google.generativeai as genai

genai.configure(api_key="your_google_api_keys")

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)