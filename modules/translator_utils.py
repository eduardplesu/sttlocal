import os
import requests

def translate_transcription_segments(segments: list[dict], target_language: str, source_language: str = None) -> list[dict]:
    """
    Translates the text of each transcription segment using the Azure Translator container.
    
    The function preserves the diarization segmentation by processing a list of segments.
    It sends a batch translation request to the translator endpoint and adds a new key 'translated_text'
    to each segment with the translated output.
    
    Parameters:
      - segments (list[dict]): List of transcription segments. Each segment should have a 'text' key.
      - target_language (str): The language code to translate the segments into (e.g., "en", "ro").
      - source_language (str): Optional source language code. If None or "auto"/"auto-detect", auto-detection is used.
    
    Returns:
      - list[dict]: Updated segments with an added 'translated_text' key.
    
    Raises:
      - Exception: If the translator key is not set or the translation API call fails.
    """
    # Retrieve the translator endpoint and key from environment variables.
    # Default endpoint is assumed to be the local container endpoint.
    endpoint = os.getenv("TRANSLATOR_ENDPOINT", "http://azure-ai-translator:5000")
    translator_key = os.getenv("TRANSLATOR_KEY")
    if not translator_key:
        raise Exception("Translator key is not set in the environment.")
    
    # Build the URL for text translation.
    # Using the Translator Text API v3.0 format.
    url = f"{endpoint}/translate?api-version=3.0&to={target_language}"
    if source_language and source_language.lower() not in ["auto", "auto-detect"]:
        url += f"&from={source_language}"
    
    # Build the payload as a list of objects, each containing the "Text" field.
    payload = [{"Text": seg.get("text", "")} for seg in segments]
    
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": translator_key
    }
    
    # Send the POST request to the translation endpoint.
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    # Parse the JSON response.
    # The expected format is a list where each element corresponds to an input text:
    # [{
    #    "translations": [{
    #         "text": "translated text",
    #         "to": "target_language"
    #     }]
    # }, ...]
    translations = response.json()
    
    # Update each segment with the translated text.
    for i, seg in enumerate(segments):
        try:
            translated_text = translations[i]["translations"][0]["text"]
        except (IndexError, KeyError):
            translated_text = ""
        seg["translated_text"] = translated_text
    
    return segments
