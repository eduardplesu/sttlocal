# Azure AI Local Speech Transcription & Translation Demo

This project demonstrates how to use local Azure Speech containers for audio transcription with diarization, translation using a local Translator container, analysis via Azure OpenAI / Phi4, and DOCX export of the transcription. The solution is built in Python with Streamlit for a friendly web interface.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Build & Deployment with Docker](#build--deployment-with-docker)
- [Local Development](#local-development)
- [Running Tests](#running-tests)
- [License](#license)

## Features

- **Speech-to-Text with Diarization:** Uses local Azure Speech containers to transcribe audio with multi-speaker support.
- **Translation:** Translates transcribed segments through a local Translator container.
- **Text Analysis & Cleaning:** Analyzes and cleans transcription text using Azure OpenAI or an alternative Phi4 model.
- **DOCX Export:** Exports transcription results with speaker names and time details to a DOCX file.
- **Web Interface:** Streamlit-based application for upload, review, edit, translation, analysis, and export functionalities.

## Project Structure

├── app.py # Main Streamlit application. ├── check_diar_properties.py # Script to list diarization properties. ├── docker/ │   ├── .env # Environment variables. │   ├── docker-compose.yml # Docker Compose configuration. │   └── Dockerfile # Dockerfile for building Streamlit app container. ├── modules/ │   ├── __init__.py
│   ├── audio_utils.py # Audio conversion helper. │   ├── docx_export.py # DOCX export functionality. │   ├── openai_analysis.py # Transcription analysis using OpenAI/Phi4. │   ├── speech_to_text.py # Speech-to-text transcription with diarization. │   ├── text_cleaning.py # Transcribed text cleaning helper. │   └── translator_utils.py # Translation of transcription segments. ├── requirements.txt # Python package dependencies. ├── test_diarization.py # Test script for diarization. ├── test_diarization_local.py # Test script for local diarization. ├── test_diarization_local_en.py# Test script for English diarization. ├── test_stt_local.py # Test script for local speech transcription.


## Prerequisites

- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/) installed.
- Python 3.12+ (if running without Docker).
- Environment variables (see next section) correctly set.

## Environment Configuration

Configuration variables are set in `docker/.env`:

- `SPEECH_ENDPOINT` and `SPEECH_KEY`: Credentials for Azure Speech Service.
- `TRANSLATOR_ENDPOINT` and `TRANSLATOR_KEY`: Credentials for the Translator API.
- `OPENAI_ENDPOINT` and `OPENAI_API_KEY`: Azure OpenAI endpoint and key.
- `PHI4_ENDPOINT` and `PHI4_KEY`: Phi4 model endpoint and key.

You can update these values in the [docker/.env](docker/.env) file.

## Build & Deployment with Docker

This project uses Docker Compose to deploy multiple containers: local STT containers for different languages, local Translator container, and the Streamlit app.

1. **Build and Run Containers:**

   Open your terminal in the project's root and run:
   ```sh
   docker-compose -f docker/docker-compose.yml up --build


   Access the Application:

The Streamlit app is exposed on port 8601. Open your browser and navigate to:

http://localhost:8601

Follow on-screen instructions to upload an audio file and step through transcription, translation, analysis, and export.

Local Development
If you want to run the application locally without Docker:

Install dependencies:

pip install -r requirements.txt

Ensure environment variables are set (e.g., in a .env file in the project root).

Run the Streamlit app:

streamlit run app.py

License
This project is provided as-is. Please refer to the appropriate licenses for individual dependencies.

Happy transcribing!