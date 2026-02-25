"""
Local Speech-to-Text (Azure Speech Container) with diarization.

Key goals:
- Feed WAV PCM into a PushAudioInputStream at a controlled pace (prevents buffering issues).
- Auto-restart on common container hiccups (buffer exceeded / websocket drops / hangs).
- Deduplicate segments across overlap windows.
"""

import os
import threading
import time
import wave
from typing import Optional, TypedDict

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

from modules.audio_utils import convert_audio_to_wav

load_dotenv()


# ---- Tuning knobs (override via env) -----------------------------------------

DISPLAY_EVERY_S = float(os.getenv("LOCAL_STT_DISPLAY_EVERY_S", "10"))
HANG_TIMEOUT_S = float(os.getenv("LOCAL_STT_HANG_TIMEOUT_S", "45"))
MAX_RESTARTS = int(os.getenv("LOCAL_STT_MAX_RESTARTS", "8"))

CHUNK_MS = int(os.getenv("LOCAL_STT_CHUNK_MS", "100"))
FEED_SPEED = float(os.getenv("LOCAL_STT_FEED_SPEED", "1.05"))

OVERLAP_S = float(os.getenv("LOCAL_STT_OVERLAP_S", "1.5"))
DEDUP_EPS_S = float(os.getenv("LOCAL_STT_DEDUP_EPS_S", "0.08"))


class TranscriptSegment(TypedDict):
    speaker_id: Optional[str]
    text: str
    offset: int  # 100ns units
    duration: int  # 100ns units


def _wav_info(path: str) -> dict:
    """Return basic WAV metadata for PCM feeding."""
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        bits_per_sample = wf.getsampwidth() * 8

    return {
        "frames": frames,
        "rate": rate,
        "channels": channels,
        "bits_per_sample": bits_per_sample,
        "duration": (frames / float(rate)) if rate else 0.0,
    }


class _WavFeeder(threading.Thread):
    """Feeds WAV PCM bytes into a PushAudioInputStream at a controlled pace."""

    def __init__(
        self,
        wav_path: str,
        push_stream: speechsdk.audio.PushAudioInputStream,
        start_frame: int,
        chunk_frames: int,
        sleep_per_chunk_s: float,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.wav_path = wav_path
        self.push_stream = push_stream
        self.start_frame = start_frame
        self.chunk_frames = chunk_frames
        self.sleep_per_chunk_s = sleep_per_chunk_s
        self.stop_event = stop_event

        self.frames_sent = 0
        self.finished = False
        self.error: Optional[Exception] = None

    def run(self) -> None:
        try:
            with wave.open(self.wav_path, "rb") as wf:
                wf.setpos(self.start_frame)

                bytes_per_frame = wf.getsampwidth() * wf.getnchannels()
                bytes_per_frame = max(bytes_per_frame, 1)

                while not self.stop_event.is_set():
                    data = wf.readframes(self.chunk_frames)
                    if not data:
                        self.finished = True
                        break

                    self.push_stream.write(data)

                    read_frames = len(data) // bytes_per_frame
                    self.frames_sent += read_frames

                    time.sleep(self.sleep_per_chunk_s)

        except Exception as exc:
            # Any runtime feeder error should trigger a restart upstream.
            self.error = exc
        finally:
            try:
                self.push_stream.close()
            except Exception:
                pass


def create_local_speech_config(language: str) -> speechsdk.SpeechConfig:
    """
    Creates a SpeechConfig for the local Azure Speech container, using ws:// endpoints
    configured via environment variables.
    """
    endpoint_map = {
        "en-US": os.getenv("LOCAL_SPEECH_ENDPOINT_EN"),
        "ro-RO": os.getenv("LOCAL_SPEECH_ENDPOINT_RO"),
        "ru-RU": os.getenv("LOCAL_SPEECH_ENDPOINT_RU"),
        "zh-CN": os.getenv("LOCAL_SPEECH_ENDPOINT_ZH"),
        "ar-AE": os.getenv("LOCAL_SPEECH_ENDPOINT_AR"),
    }

    host_endpoint = endpoint_map.get(language)
    if not host_endpoint:
        raise ValueError(f"Unsupported language or missing endpoint: {language}")

    full_endpoint = (
        f"{host_endpoint}/speech/recognition/dictation/cognitiveservices/v1"
    )
    print(
        f"[DEBUG] Using local STT endpoint for '{language}': "
        f"host={host_endpoint}, endpoint={full_endpoint}"
    )

    speech_config = speechsdk.SpeechConfig(host=host_endpoint, endpoint=full_endpoint)
    speech_config.speech_recognition_language = language
    speech_config.set_profanity(speechsdk.ProfanityOption.Raw)

    # Optional: some container setups still expect region/key properties.
    region = os.getenv("SPEECH_REGION") or os.getenv("region")
    if region:
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_Region, region
        )

    container_key = os.getenv("SPEECH_KEY") or os.getenv("apiKey")
    if container_key:
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_Key, container_key
        )

    # Diarization (if supported by your container).
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
        "true",
    )

    # Silence timeouts (ms)
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        "120000",
    )
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
        "120000",
    )

    return speech_config


def transcribe_with_diarization_local(file_path: str, language: str) -> list[TranscriptSegment]:
    """
    Transcribes an audio file with diarization using the local Azure Speech container.

    Returns:
        [
          {"speaker_id": str|None, "text": str, "offset": int, "duration": int},
          ...
        ]
    """
    wav_path = convert_audio_to_wav(file_path)
    speech_config = create_local_speech_config(language)

    wav = _wav_info(wav_path)
    total_frames = int(wav["frames"])
    rate = int(wav["rate"])
    channels = int(wav["channels"])
    bits_per_sample = int(wav["bits_per_sample"])
    wav_duration = float(wav["duration"])

    if rate <= 0 or channels <= 0 or bits_per_sample <= 0:
        raise ValueError(f"Invalid WAV metadata for: {wav_path}")

    chunk_frames = max(1, int(rate * (CHUNK_MS / 1000.0)))
    sleep_per_chunk_s = (CHUNK_MS / 1000.0) / max(FEED_SPEED, 0.1)

    scale_100ns = 10_000_000

    transcription_results: list[TranscriptSegment] = []
    accepted_last_end_s = -1.0

    cursor_frame = 0
    restarts = 0
    backoff_s = 1.0

    while cursor_frame < total_frames:
        session_idx = restarts
        session_base_s = cursor_frame / float(rate)

        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=rate,
            bits_per_sample=bits_per_sample,
            channels=channels,
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        done = False
        restart_needed = False
        restart_reason: Optional[str] = None

        last_activity = time.time()
        last_print = 0.0

        stop_event = threading.Event()
        feeder = _WavFeeder(
            wav_path=wav_path,
            push_stream=push_stream,
            start_frame=cursor_frame,
            chunk_frames=chunk_frames,
            sleep_per_chunk_s=sleep_per_chunk_s,
            stop_event=stop_event,
        )

        def _safe_lower(s: Optional[str]) -> str:
            return (s or "").lower()

        def transcribed_cb(evt) -> None:
            nonlocal accepted_last_end_s, last_activity
            last_activity = time.time()

            result = getattr(evt, "result", None)
            if not result or getattr(result, "reason", None) != speechsdk.ResultReason.RecognizedSpeech:
                return

            text = (getattr(result, "text", "") or "").strip()
            if not text:
                return

            offset_100ns = int(getattr(result, "offset", 0) or 0)
            duration_100ns = int(getattr(result, "duration", 0) or 0)

            seg_start_s = (offset_100ns / scale_100ns) + session_base_s
            seg_end_s = ((offset_100ns + duration_100ns) / scale_100ns) + session_base_s

            # Deduplicate across overlaps
            if seg_end_s <= accepted_last_end_s + DEDUP_EPS_S:
                return

            speaker_id = getattr(result, "speaker_id", None)

            transcription_results.append(
                {
                    "speaker_id": speaker_id,
                    "text": text,
                    "offset": int(seg_start_s * scale_100ns),
                    "duration": int((seg_end_s - seg_start_s) * scale_100ns),
                }
            )

            accepted_last_end_s = max(accepted_last_end_s, seg_end_s)
            print(f"[DEBUG] TRANSCRIBED s{session_idx} {speaker_id}: {text}")

        def transcribing_cb(evt) -> None:
            nonlocal last_activity, last_print
            last_activity = time.time()

            now = time.time()
            if now - last_print < DISPLAY_EVERY_S:
                return

            result = getattr(evt, "result", None)
            text = (getattr(result, "text", "") or "").strip() if result else ""
            speaker_id = getattr(result, "speaker_id", None) if result else None

            if text:
                print(f"[DEBUG] TRANSCRIBING s{session_idx} {speaker_id}: {text[:140]}")
            last_print = now

        def canceled_cb(evt) -> None:
            nonlocal done, restart_needed, restart_reason

            # Different event shapes exist; be defensive.
            result = getattr(evt, "result", None)
            cancellation_details = getattr(result, "cancellation_details", None) if result else None
            error_details = getattr(cancellation_details, "error_details", None) if cancellation_details else None
            msg = _safe_lower(error_details)

            if "client buffer exceeded" in msg:
                restart_needed = True
                restart_reason = "client_buffer_exceeded"
            elif "websocket" in msg or "connection" in msg or "timeout" in msg:
                restart_needed = True
                restart_reason = "connection_or_timeout"
            else:
                restart_reason = error_details or "canceled"

            done = True

        def session_stopped_cb(_evt) -> None:
            nonlocal done
            done = True

        conversation_transcriber.transcribed.connect(transcribed_cb)
        conversation_transcriber.transcribing.connect(transcribing_cb)
        conversation_transcriber.session_stopped.connect(session_stopped_cb)
        conversation_transcriber.canceled.connect(canceled_cb)

        feeder.start()
        conversation_transcriber.start_transcribing_async().get()

        while not done:
            time.sleep(0.5)
            if time.time() - last_activity > HANG_TIMEOUT_S:
                restart_needed = True
                restart_reason = "hang_timeout"
                done = True

        stop_event.set()
        try:
            conversation_transcriber.stop_transcribing_async().get()
        except Exception:
            pass

        feeder.join(timeout=5)

        # Advance cursor by what we actually fed this session
        cursor_frame = min(total_frames, cursor_frame + int(feeder.frames_sent))

        # Decide whether to restart
        if feeder.error and not restart_needed:
            restart_needed = True
            restart_reason = f"feeder_error:{type(feeder.error).__name__}"

        if not restart_needed and cursor_frame < total_frames and not feeder.finished:
            restart_needed = True
            restart_reason = "session_stopped_early"

        if restart_needed:
            restarts += 1
            if restarts > MAX_RESTARTS:
                raise RuntimeError(
                    f"Too many restarts ({MAX_RESTARTS}). Last reason: {restart_reason}"
                )

            overlap_frames = int(OVERLAP_S * rate)
            cursor_frame = max(0, cursor_frame - overlap_frames)

            print(
                f"[DEBUG] Restarting transcription (reason={restart_reason}). "
                f"cursor={cursor_frame / float(rate):.2f}s backoff={backoff_s:.1f}s"
            )

            time.sleep(backoff_s)
            backoff_s = min(20.0, backoff_s * 2.0)
            continue

        # If the container naturally segments sessions, loop continues until EOF.
        if cursor_frame < total_frames:
            continue

        break

    if not transcription_results:
        print("[DEBUG] No results captured. Consider checking local speech container logs.")
    elif accepted_last_end_s > 0 and wav_duration > 0 and accepted_last_end_s < wav_duration * 0.98:
        print(
            f"[DEBUG] WARNING: transcript may end early "
            f"({accepted_last_end_s:.2f}s vs WAV {wav_duration:.2f}s)."
        )

    return transcription_results