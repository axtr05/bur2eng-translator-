"""
Burmese speech -> English speech translator (fully offline pipeline)
Gradio web interface

Pipeline:
  1. Record/upload Burmese speech (via browser mic or file upload)
  2. Silero VAD trims out silence
  3. faster-whisper transcribes the Burmese speech (models/whisper-large-v3-myanmar)
  4. NLLB-200 translates the Burmese text into English
  5. MMS-TTS (facebook/mms-tts-eng) synthesizes English audio locally

OFFLINE NOTES
-------------
Every model here (Silero VAD, faster-whisper, NLLB, MMS-TTS) runs 100% locally
once its weights are downloaded and cached. There is no step that calls a
remote API at inference time.

The very first time you run this on a machine, each model needs to download
its weights from the Hugging Face Hub / torch.hub (one-time, requires
internet). After that, everything is read from the local cache (usually
~/.cache/huggingface and ~/.cache/torch) and works with no network
connection at all.

To force strict offline mode (fail fast instead of hanging if weights
aren't cached), set these env vars before running:

    # Linux/Mac
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    # Windows PowerShell
    $env:HF_HUB_OFFLINE=1
    $env:TRANSFORMERS_OFFLINE=1

RUN
---
    pip install gradio torch numpy faster-whisper transformers silero-vad soundfile
    python b2e.py

This launches a local web server (default http://127.0.0.1:7860) with a
microphone/file input, and text + audio outputs.
"""

import os
import time

import torch
import numpy as np
import gradio as gr

from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

from silero_vad import load_silero_vad, get_speech_timestamps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Uncomment to hard-enforce offline mode (fails loudly if a model isn't
# already cached, instead of trying to hit the network):
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# faster-whisper model path
WHISPER_MODEL_SIZE = "/home/axtr/Projects/bur_eng_translate/models/whisper-small-myanmar-ct2-int8fp16"
WHISPER_SIZE_CHOICES = ["/home/axtr/Projects/bur_eng_translate/models/whisper-small-myanmar-ct2-int8fp16"]

# Cap translation length — short utterances don't need 256 new tokens, and
# capping this cuts NLLB generation time noticeably for typical sentences.
NLLB_MAX_NEW_TOKENS = 128

# Use int8 dynamic quantization for NLLB on CPU (roughly 1.5-2.5x faster
# generation on CPU with a small, usually imperceptible, quality tradeoff).
QUANTIZE_NLLB_ON_CPU = True

# Smaller NLLB variant = much faster translation, small accuracy tradeoff.
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# Local, offline English TTS model (Meta MMS project).
MMS_TTS_MODEL_ID = "facebook/mms-tts-eng"

SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.3 # lower = more lenient (default Silero threshold is 0.5)


# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------

print("Loading Silero VAD...")
vad_model = load_silero_vad()

WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
print(f"Using Whisper device: {WHISPER_DEVICE} (Whisper compute type: {WHISPER_COMPUTE_TYPE})")

OTHER_DEVICE = "cpu"
NUM_CPU_THREADS = os.cpu_count() or 4
torch.set_num_threads(NUM_CPU_THREADS)
print(f"CPU mode: using {NUM_CPU_THREADS} threads for torch ops.")

# faster-whisper models are loaded lazily and cached per size, so switching
# in the UI is instant after the first use.
_whisper_models = {}


def get_whisper_model(size):
    if size not in _whisper_models:
        print(f"Loading faster-whisper '{size}' model...")
        _whisper_models[size] = WhisperModel(
            size,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=0,
        )
    return _whisper_models[size]


# Preload the default size at startup so the first request is fast.
get_whisper_model(WHISPER_MODEL_SIZE)

print("Loading NLLB tokenizer + model...")
nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID).to(OTHER_DEVICE)
nllb_model.eval()

if QUANTIZE_NLLB_ON_CPU:
    print("Quantizing NLLB model to int8 for faster CPU inference...")
    nllb_model = torch.quantization.quantize_dynamic(
        nllb_model, {torch.nn.Linear}, dtype=torch.qint8
    )

print("Loading local English TTS model (MMS)...")
mms_tokenizer = AutoTokenizer.from_pretrained(MMS_TTS_MODEL_ID)
mms_model = VitsModel.from_pretrained(MMS_TTS_MODEL_ID).to(OTHER_DEVICE)
mms_model.eval()
MMS_SAMPLE_RATE = mms_model.config.sampling_rate

print("\nModels ready. (fully offline mode - no network calls happen after this point)\n")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def resample_if_needed(audio, orig_sr, target_sr=SAMPLE_RATE):
    """Very light resampling using linear interpolation (avoids extra deps
    like librosa/torchaudio). Fine for speech; Gradio mic input is usually
    already 16/44/48kHz mono float."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    x_old = np.linspace(0, duration, num=len(audio))
    x_new = np.linspace(0, duration, num=target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def to_mono_float32(audio):
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    # Gradio mic input can come back as int16 PCM; normalize if so.
    if np.abs(audio).max() > 1.5:
        audio = audio / 32768.0
    return audio


def vad_filter(audio):
    """Keep only the speech segments detected by Silero VAD."""
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak < 0.01:
        return None, f"WARNING: audio is nearly silent (peak={peak:.4f}). Check your mic."

    audio_tensor = torch.tensor(audio)
    speech = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
    )

    if len(speech) == 0:
        return None, "No speech detected."

    speech_chunks = [audio[s["start"]:s["end"]] for s in speech]
    return np.concatenate(speech_chunks), None


def transcribe_burmese(audio, whisper_size):
    """faster-whisper: Burmese audio -> Burmese text."""
    model = get_whisper_model(whisper_size)
    segments, _info = model.transcribe(
        audio,
        language="my",
        task="transcribe",
        beam_size=1,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return "".join(segment.text for segment in segments).strip()

def translate_to_english(text):
    """NLLB: Burmese text -> English text."""
    nllb_tokenizer.src_lang = "mya_Mymr"

    inputs = nllb_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(OTHER_DEVICE)

    target_lang_id = nllb_tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        translated_ids = nllb_model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_new_tokens=NLLB_MAX_NEW_TOKENS,
            num_beams=1,
        )

    return nllb_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]


def speak_english(text):
    """Synthesize English text to speech locally (MMS-TTS). Returns
    (sample_rate, waveform) tuple, the format Gradio's Audio output expects."""
    inputs = mms_tokenizer(text, return_tensors="pt").to(OTHER_DEVICE)

    with torch.no_grad():
        output = mms_model(**inputs).waveform

    waveform = output.squeeze().cpu().numpy().astype(np.float32)
    return MMS_SAMPLE_RATE, waveform


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def run_pipeline(mic_audio, file_audio, whisper_size, progress=gr.Progress()):
    """mic_audio / file_audio are (sample_rate, np.ndarray) tuples from Gradio,
    or None. Mic takes priority if both are provided."""
    source = mic_audio if mic_audio is not None else file_audio

    if source is None:
        return "No audio provided.", "", None, ""

    orig_sr, raw_audio = source
    log_lines = [f"Whisper model: {whisper_size}"]

    progress(0.1, desc="Preparing audio...")
    audio = to_mono_float32(raw_audio)
    audio = resample_if_needed(audio, orig_sr, SAMPLE_RATE)
    log_lines.append(f"Recorded duration: {len(audio) / SAMPLE_RATE:.2f}s (resampled from {orig_sr}Hz)")

    progress(0.25, desc="Running voice activity detection...")
    speech_audio, vad_error = vad_filter(audio)
    if vad_error:
        log_lines.append(vad_error)
        return "", "", None, "\n".join(log_lines)
    log_lines.append(f"Speech after VAD: {len(speech_audio) / SAMPLE_RATE:.2f}s")

    progress(0.45, desc=f"Transcribing Burmese speech...")
    start = time.time()
    burmese_text = transcribe_burmese(speech_audio, whisper_size)
    log_lines.append(f"Transcription time: {time.time() - start:.2f}s")

    if not burmese_text.strip():
        log_lines.append("Empty transcription, stopping.")
        return "", "", None, "\n".join(log_lines)

    progress(0.65, desc="Translating to English...")
    start = time.time()
    english_text = translate_to_english(burmese_text)
    log_lines.append(f"Translation time: {time.time() - start:.2f}s")

    if not english_text.strip():
        log_lines.append("Translation failed (empty output).")
        return burmese_text, "", None, "\n".join(log_lines)

    progress(0.85, desc="Synthesizing English speech...")
    start = time.time()
    sr, waveform = speak_english(english_text)
    log_lines.append(f"TTS time: {time.time() - start:.2f}s")

    progress(1.0, desc="Done")
    return burmese_text, english_text, (sr, waveform), "\n".join(log_lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Burmese -> English Offline Speech Translator") as demo:
    gr.Markdown(
        """
        # Burmese -> English Offline Speech Translator
        Speak (or upload) Burmese audio. It's transcribed with **faster-whisper**,
        translated with **NLLB-200**, and spoken back in English with **MMS-TTS** —
        all running locally, no external API calls at inference time.
        """
    )

    with gr.Row():
        with gr.Column():
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Burmese Speech")
            file_input = gr.Audio(sources=["upload"], type="numpy", label="...or upload a Burmese audio file")
            whisper_size_dd = gr.Dropdown(
                choices=WHISPER_SIZE_CHOICES,
                value=WHISPER_MODEL_SIZE,
                label="Whisper model path",
                info="First use loads it (one-time delay); cached after that.",
            )
            run_btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            burmese_out = gr.Textbox(label="Burmese Transcript", lines=3)
            english_out = gr.Textbox(label="Translated English", lines=3)
            audio_out = gr.Audio(label="English Speech", type="numpy")
            log_out = gr.Textbox(label="Log", lines=5)

    run_btn.click(
        fn=run_pipeline,
        inputs=[mic_input, file_input, whisper_size_dd],
        outputs=[burmese_out, english_out, audio_out, log_out],
    )

if __name__ == "__main__":
    demo.queue().launch()
