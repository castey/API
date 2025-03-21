from api import chat_with_model, SUPPORTED_MODELS

messages = [
    {"role": "system", "content": "You are a helpful assistant named Bob who always introduces himself as Bob."},
    {"role": "user", "content": "tell me a fun fact about knowledge graphs?"}
]

options = {
    "temperature": 1.0,
    "top_p": 0.9,
    "presence_penalty": 1.2,
    "num_predict": 50 # max output tokens
}

for model in SUPPORTED_MODELS:
    print(f"\n===== Testing model: {model} =====")
    try:
        response = chat_with_model(
            model_name=model, 
            messages=messages, 
            options=options
        ) 

        message = response["message"]["content"].strip()
        token_count = response.get("eval_count", "N/A")
        total_duration_ms = round(response.get("total_duration", 0) / 1e6, 2)
        load_duration_ms = round(response.get("load_duration", 0) / 1e6, 2)

        print(f"\n📦 Model: {model}")
        print(f"🕒 Total Duration: {total_duration_ms} ms (Load: {load_duration_ms} ms)")
        print(f"🔤 Tokens Generated: {token_count}")
        print("\n💬 Response:\n" + "-"*50)
        print(message)
        print("-"*50)

    except Exception as e:
        print(f"❌ Error with model {model}: {e}")



