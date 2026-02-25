import streamlit as st
import os
from datetime import datetime

# Local Speech-to-Text function
from modules.speech_to_text import transcribe_with_diarization_local
# Translator utility
from modules.translator_utils import translate_transcription_segments
# DOCX export function
from modules.docx_export import export_transcription_to_docx
# Analysis function using Azure OpenAI or Phi4
from modules.openai_analysis import analyze_transcription
# Text cleaning function (via OpenAI or Phi4)
from modules.text_cleaning import clean_segments_with_openai

# Clear session state for new upload
def clear_previous_session():
    keys_to_clear = [
        "temp_file_path",
        "transcription_results",
        "uploaded_filename",
        "analysis_result",
        "cleaned_transcription",
        "translated_transcription"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Tab 1: Upload & Transcribe
def upload_and_transcribe():
    st.header("1. Upload & Transcribe")
    uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["mp3", "wav"], key="upload")
    if uploaded_file is not None:
        if st.session_state.get("uploaded_filename") != uploaded_file.name:
            clear_previous_session()
            st.session_state.uploaded_filename = uploaded_file.name
        if not st.session_state.get("temp_file_path"):
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.temp_file_path = temp_file_path
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        language_options = ["ro-RO", "en-US", "ru-RU", "zh-CN", "ar-AE"]
        language_selected = st.selectbox("Select transcription language:", language_options, index=0)
        if st.button("Start Transcription", key="transcribe_button"):
            if not st.session_state.get("temp_file_path"):
                st.error("No file available for transcription. Please upload an audio file.")
                return
            with st.spinner("Transcribing using local container..."):
                try:
                    transcription_results = transcribe_with_diarization_local(
                        st.session_state.temp_file_path,
                        language=language_selected
                    )
                    st.session_state.transcription_results = transcription_results
                    st.success("Transcription completed!")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
    else:
        st.info("Please upload an audio file.")

# Tab 2: Review & Edit
def review_and_edit():
    st.header("2. Review & Edit")
    if not st.session_state.get("temp_file_path"):
        st.warning("No audio file uploaded yet. Please complete step 1.")
        return
    st.audio(st.session_state.temp_file_path, format="audio/wav")
    if not st.session_state.get("transcription_results"):
        st.warning("No transcription results available. Please complete transcription first.")
        return
    st.subheader("Edit Transcription Segments")
    with st.form("edit_transcription_form"):
        edited_transcriptions = []
        for i, segment in enumerate(st.session_state.transcription_results):
            speaker = segment.get("speaker_id", "Unknown")
            text = segment.get("text", "")
            new_text = st.text_area(label=f"Segment {i+1} - Speaker {speaker}",
                                    value=text,
                                    key=f"segment_{i}")
            edited_transcriptions.append({**segment, "text": new_text})
        # Select cleaning engine: gpt4o or phi4
        cleaning_engine = st.selectbox("Select Text Cleaning Engine:", options=["gpt4o", "phi4"], index=0)
        col1, col2 = st.columns(2)
        with col1:
            save_edits = st.form_submit_button("Save Edits")
        with col2:
            clean_segments = st.form_submit_button("Clean All Segments")
        if save_edits:
            st.session_state.transcription_results = edited_transcriptions
            st.success("Transcription edits saved!")
        if clean_segments:
            try:
                cleaned_transcriptions = clean_segments_with_openai(edited_transcriptions, engine=cleaning_engine)
                st.session_state.transcription_results = cleaned_transcriptions
                st.session_state.cleaned_transcription = "\n".join([seg["text"] for seg in cleaned_transcriptions])
                st.success("All segments cleaned!")
            except Exception as e:
                st.error(f"Text cleaning failed: {e}")
    st.subheader("Assign Speaker Names")
    with st.form("assign_names_form"):
        speaker_names = {}
        unique_speakers = {seg.get("speaker_id", "Unknown") for seg in st.session_state.transcription_results}
        for speaker in unique_speakers:
            name = st.text_input(label=f"Name for Speaker {speaker}",
                                 value=f"Speaker {speaker}",
                                 key=f"name_{speaker}")
            speaker_names[speaker] = name
        if st.form_submit_button("Save Speaker Names"):
            for seg in st.session_state.transcription_results:
                seg["speaker_name"] = speaker_names.get(seg.get("speaker_id", "Unknown"), "Unknown")
            st.success("Speaker names saved!")

# Tab 3: Translate Transcript
def translate_transcript():
    st.header("3. Translate Transcript")
    if not st.session_state.get("transcription_results"):
        st.warning("No transcription available. Please complete transcription first.")
        return
    segments = st.session_state.transcription_results
    with st.form("translate_form"):
        target_language = st.selectbox("Select target language:", ["en", "ro", "ru", "zh", "ar"], key="target_lang")
        source_language = st.selectbox("Select source language:", ["auto", "ro", "en", "ru", "zh", "ar"], key="source_lang")
        if st.form_submit_button("Translate Transcript"):
            with st.spinner("Translating via local translator container..."):
                try:
                    used_source = None if source_language.lower() in ["auto", "auto-detect"] else source_language
                    translated_segments = translate_transcription_segments(
                        segments,
                        target_language=target_language,
                        source_language=used_source
                    )
                    translated_text = "\n\n".join([
                        f"Speaker {seg.get('speaker_name', seg.get('speaker_id', 'Unknown'))}: {seg.get('translated_text', '')}"
                        for seg in translated_segments
                    ])
                    st.session_state.translated_transcription = translated_text
                    st.success("Translation completed!")
                except Exception as e:
                    st.error(f"Translation failed: {e}")
    if st.session_state.get("translated_transcription"):
        st.subheader("Translated Transcript")
        st.text_area("Translated Transcript", st.session_state.translated_transcription, height=300)

# Tab 4: Analysis
def analysis_tab():
    st.header("4. Analysis")
    if not st.session_state.get("transcription_results"):
        st.warning("No transcription available for analysis. Please complete transcription and editing first.")
        return
    transcription_text = "\n".join([seg.get("text", "") for seg in st.session_state.transcription_results])
    # Select analysis engine: gpt4o or phi4
    analysis_engine = st.selectbox("Select Analysis Engine:", options=["gpt4o", "phi4"], index=0)
    if st.button("Analyze Transcription", key="analyze_button"):
        with st.spinner("Analyzing transcription..."):
            try:
                analysis_result = analyze_transcription(transcription_text, engine=analysis_engine)
                st.session_state.analysis_result = analysis_result
                st.success("Analysis completed!")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    if st.session_state.get("analysis_result"):
        st.subheader("Analysis Output")
        st.text_area("Analysis", st.session_state.analysis_result, height=300)
        if st.button("Save Analysis", key="save_analysis"):
            st.success("Analysis saved!")
    else:
        st.info("Run analysis to see results.")

# Tab 5: Export & Save
def export_and_save():
    st.header("5. Export & Save")
    if not st.session_state.get("transcription_results"):
        st.warning("No transcription available. Please complete transcription and editing first.")
        return
    analysis_text = st.session_state.get("analysis_result")
    final_transcription = st.session_state.transcription_results
    if st.button("Generate DOCX and Download", key="download_button"):
        with st.spinner("Generating DOCX..."):
            try:
                unique_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
                output_filename = f"transcription_{unique_suffix}.docx"
                docx_file = export_transcription_to_docx(
                    final_transcription,
                    analysis_text=analysis_text,
                    translated_transcription=st.session_state.get("translated_transcription"),
                    cleaned_transcription=st.session_state.get("cleaned_transcription"),
                    output_filename=output_filename
                )
                with open(docx_file, "rb") as f:
                    st.download_button("Download DOCX", data=f.read(), file_name=docx_file)
                st.success("DOCX generated!")
            except Exception as e:
                st.error(f"Error generating DOCX: {e}")
    st.markdown("**Note:** In production, implement cleanup for temporary files.")

# Main App with Tabs
def main():
    st.title("Azure AI Local Speech Transcription & Translation Demo")
    st.markdown(
        "This demo showcases local Azure Speech containers for transcription, a local translator container for translation, "
        "plus analysis with Azure OpenAI and text cleaning. Use the tabs below to progress through each step."
    )
    tabs = st.tabs([
        "Upload & Transcribe",
        "Review & Edit",
        "Translate Transcript",
        "Analysis",
        "Export & Save"
    ])
    with tabs[0]:
        upload_and_transcribe()
    with tabs[1]:
        review_and_edit()
    with tabs[2]:
        translate_transcript()
    with tabs[3]:
        analysis_tab()
    with tabs[4]:
        export_and_save()

if __name__ == "__main__":
    for key in [
        "transcription_results",
        "temp_file_path",
        "uploaded_filename",
        "analysis_result",
        "cleaned_transcription",
        "translated_transcription"
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    main()
