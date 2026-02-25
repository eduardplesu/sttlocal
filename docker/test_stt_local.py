import time
import azure.cognitiveservices.speech as speechsdk

def test_stt_local():
    """
    Continuously transcribe an audio file with a local STT container.
    Prints recognized text or cancellation details.
    """

    # 1. Point to your local container endpoint (mapped from port 5000).
    speech_config = speechsdk.SpeechConfig(host="ws://localhost:5103")
    
    # 2. Set the recognition language to Romanian (adjust as needed).
    speech_config.speech_recognition_language = "ro-RO"

    # 3. Configure the audio file to transcribe.
    audio_config = speechsdk.audio.AudioConfig(filename="stenograma.wav")

    # 4. Create a standard SpeechRecognizer for continuous recognition.
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # 5. A simple flag to know when we're done.
    done = False

    # --- EVENT CALLBACKS ---

    # Recognizing: intermediate results (not final yet).
    def recognizing_cb(evt: speechsdk.SessionEventArgs):
        # You can uncomment to see partial results if needed.
        # print(f"[RECOGNIZING] {evt.result.text}")
        pass

    # Recognized: final results (one utterance recognized).
    def recognized_cb(evt: speechsdk.SessionEventArgs):
        result = evt.result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"[RECOGNIZED] {result.text}")
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("[NO MATCH] The speech could not be recognized.")

    # Canceled: an error, or user-initiated stop.
    def canceled_cb(evt: speechsdk.SessionEventArgs):
        result = evt.result
        print("[CANCELED]")
        if result.cancellation_details:
            details = result.cancellation_details
            print(f"  Reason: {details.reason}")
            print(f"  Error Details: {details.error_details}")
        nonlocal done
        done = True

    # Session started: the recognition session (not the first utterance).
    def session_started_cb(evt: speechsdk.SessionEventArgs):
        print(f"[SESSION STARTED] {evt.session_id}")

    # Session stopped: no more recognition results, or an error triggered stop.
    def session_stopped_cb(evt: speechsdk.SessionEventArgs):
        print(f"[SESSION STOPPED] {evt.session_id}")
        nonlocal done
        done = True

    # --- CONNECT EVENT HANDLERS ---
    speech_recognizer.recognizing.connect(recognizing_cb)
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.canceled.connect(canceled_cb)
    speech_recognizer.session_started.connect(session_started_cb)
    speech_recognizer.session_stopped.connect(session_stopped_cb)

    # 6. Start continuous recognition.
    speech_recognizer.start_continuous_recognition_async().get()
    print("Continuous recognition started...\n")

    # 7. Keep the script running until recognition is done or canceled.
    while not done:
        time.sleep(0.5)

    # 8. Stop recognition after completion/timeout/error.
    speech_recognizer.stop_continuous_recognition_async().get()

if __name__ == "__main__":
    test_stt_local()
