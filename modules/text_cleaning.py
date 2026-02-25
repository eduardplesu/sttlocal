import os
import json

def clean_segments_with_openai(segments: list[dict], engine: str = "gpt4o") -> list[dict]:
    """
    Cleans transcribed segments using the selected engine.
    Builds a prompt with all segments and instructs the model to return a JSON array with a 'text' key.
    """
    user_content = (
        "Please clean the following transcribed segments. For each segment, remove extraneous characters, "
        "correct grammatical, punctuation, and spelling errors while preserving the original meaning. "
        "Do NOT censor or mask any words. Return the result as a JSON array of objects, each with a single key 'text'.\n\n"
    )
    for i, seg in enumerate(segments, start=1):
        user_content += f"Segment {i}: {seg.get('text', '')}\n---\n"

    system_message = "You are an AI assistant that cleans and formats transcribed text."

    if engine == "gpt4o":
        import openai
        # Configure Azure OpenAI settings
        openai.api_type = "azure"
        openai.api_base = os.getenv("OPENAI_ENDPOINT")  # e.g., https://empowergovswcent.openai.azure.com/
        openai.api_version = "2025-01-01-preview"  # Adjust if necessary
        openai.api_key = os.getenv("OPENAI_API_KEY")
        deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        response = openai.ChatCompletion.create(
            engine=deployment,
            messages=prompt,
            max_tokens=3000,
            temperature=0.5,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
        cleaned_text_json = response.choices[0].message["content"]

    elif engine == "phi4":
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        phi4_endpoint = os.getenv("PHI4_ENDPOINT")
        phi4_key = os.getenv("PHI4_KEY")
        if not phi4_endpoint or not phi4_key:
            raise Exception("Phi4 endpoint or key is not set in the environment.")
        # Remove trailing path if present
        if phi4_endpoint.endswith("/v1/chat/completions"):
            phi4_endpoint = phi4_endpoint[:-len("/v1/chat/completions")]
        client = ChatCompletionsClient(
            endpoint=phi4_endpoint,
            credential=AzureKeyCredential(phi4_key)
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 3000,
            "temperature": 0.5,
            "top_p": 0.95,
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
        response = client.complete(payload)
        cleaned_text_json = response.choices[0].message.content
    else:
        raise Exception("Unsupported engine specified. Use 'gpt4o' or 'phi4'.")

    # Remove markdown code block formatting if present.
    if cleaned_text_json.startswith("```"):
        lines = cleaned_text_json.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text_json = "\n".join(lines)

    try:
        cleaned_array = json.loads(cleaned_text_json)
        if isinstance(cleaned_array, list) and len(cleaned_array) == len(segments):
            for i, seg in enumerate(segments):
                seg["text"] = cleaned_array[i].get("text", seg.get("text", ""))
        else:
            print("Warning: Returned JSON does not match expected format or segment count.")
    except Exception as e:
        print("Error parsing JSON from cleaning API:", e)
        print("Raw response for debugging:", cleaned_text_json)

    return segments
