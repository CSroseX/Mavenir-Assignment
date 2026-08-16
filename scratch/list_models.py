from google import genai

try:
    client = genai.Client()
    print("Fetching available models from your API key...")
    models = list(client.models.list())
    
    embed_models = [m for m in models if 'embed' in m.name.lower()]
    
    if embed_models:
        print("\nFound these embedding models:")
        for m in embed_models:
            print(f"- {m.name}")
    else:
        print("\nNO embedding models found for this API key!")
        print("Here are the first 10 models you DO have access to:")
        for m in models[:10]:
            print(f"- {m.name}")
            
except Exception as e:
    print(f"Error fetching models: {e}")
