

import os
import time

import torch
import numpy as np
import gradio as gr

from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

from silero_vad import load_silero_vad, get_speech_timestamps


WHISPER_MODEL_SIZE = "tiny"
WHISPER_SIZE_CHOICES = ["tiny", "base", "small"]

NLLB_MAX_NEW_TOKENS = 128

#
QUANTIZE_NLLB_ON_CPU = True

# Smaller NLLB variant = much faster translation, small accuracy tradeoff.
# Use "facebook/nllb-200-distilled-1.3B" for higher quality.
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# Local, offline Burmese TTS model (Meta MMS project).
MMS_TTS_MODEL_ID = "facebook/mms-tts-mya"

SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.3 # lower = more lenient (default Silero threshold is 0.5)


# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------

print("Loading Silero VAD...")
vad_model = load_silero_vad()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"Using device: {DEVICE} (Whisper compute type: {WHISPER_COMPUTE_TYPE})")

NUM_CPU_THREADS = os.cpu_count() or 4
if DEVICE == "cpu":
    # Use all available cores for PyTorch (NLLB, MMS-TTS) ops.
    torch.set_num_threads(NUM_CPU_THREADS)
    print(f"CPU mode: using {NUM_CPU_THREADS} threads for torch ops.")

# faster-whisper models are loaded lazily and cached per size, so switching
# "tiny" / "base" / "small" in the UI is instant after the first use of each.
_whisper_models = {}


def get_whisper_model(size):
    if size not in _whisper_models:
        print(f"Loading faster-whisper '{size}' model...")
        _whisper_models[size] = WhisperModel(
            size,
            device=DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=NUM_CPU_THREADS if DEVICE == "cpu" else 0,
        )
    return _whisper_models[size]


# Preload the default size at startup so the first request is fast.
get_whisper_model(WHISPER_MODEL_SIZE)

print("Loading NLLB tokenizer + model...")
nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID).to(DEVICE)
nllb_model.eval()

if DEVICE == "cpu" and QUANTIZE_NLLB_ON_CPU:
    print("Quantizing NLLB model to int8 for faster CPU inference...")
    nllb_model = torch.quantization.quantize_dynamic(
        nllb_model, {torch.nn.Linear}, dtype=torch.qint8
    )

print("Loading local Burmese TTS model (MMS)...")
mms_tokenizer = AutoTokenizer.from_pretrained(MMS_TTS_MODEL_ID)
mms_model = VitsModel.from_pretrained(MMS_TTS_MODEL_ID).to(DEVICE)
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


def transcribe_english(audio, whisper_size):
    """faster-whisper: English audio -> English text."""
    model = get_whisper_model(whisper_size)
    segments, _info = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=1,
    )
    return "".join(segment.text for segment in segments).strip()


def translate_to_burmese(text):
    """NLLB: English text -> Burmese text."""
    nllb_tokenizer.src_lang = "eng_Latn"

    inputs = nllb_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    target_lang_id = nllb_tokenizer.convert_tokens_to_ids("mya_Mymr")

    with torch.no_grad():
        translated_ids = nllb_model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_new_tokens=NLLB_MAX_NEW_TOKENS,
            num_beams=1,
        )

    return nllb_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]


def speak_burmese(text):
    """Synthesize Burmese text to speech locally (MMS-TTS). Returns
    (sample_rate, waveform) tuple, the format Gradio's Audio output expects."""
    inputs = mms_tokenizer(text, return_tensors="pt").to(DEVICE)

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
    log_lines = [f"Whisper size: {whisper_size}"]

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

    progress(0.45, desc=f"Transcribing English speech ({whisper_size})...")
    start = time.time()
    english_text = transcribe_english(speech_audio, whisper_size)
    log_lines.append(f"Transcription time: {time.time() - start:.2f}s")

    if not english_text.strip():
        log_lines.append("Empty transcription, stopping.")
        return "", "", None, "\n".join(log_lines)

    progress(0.65, desc="Translating to Burmese...")
    start = time.time()
    burmese_text = translate_to_burmese(english_text)
    log_lines.append(f"Translation time: {time.time() - start:.2f}s")

    if not burmese_text.strip():
        log_lines.append("Translation failed (empty output).")
        return english_text, "", None, "\n".join(log_lines)

    progress(0.85, desc="Synthesizing Burmese speech...")
    start = time.time()
    sr, waveform = speak_burmese(burmese_text)
    log_lines.append(f"TTS time: {time.time() - start:.2f}s")

    progress(1.0, desc="Done")
    return english_text, burmese_text, (sr, waveform), "\n".join(log_lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="English -> Burmese Offline Speech Translator") as demo:
    gr.Markdown(
        """
        # English -> Burmese Offline Speech Translator
        Speak (or upload) English audio. It's transcribed with **faster-whisper**,
        translated with **NLLB-200**, and spoken back in Burmese with **MMS-TTS** —
        all running locally, no external API calls at inference time.
        """
    )

    with gr.Row():
        with gr.Column():
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record English speech")
            file_input = gr.Audio(sources=["upload"], type="numpy", label="...or upload an English audio file")
            whisper_size_dd = gr.Dropdown(
                choices=WHISPER_SIZE_CHOICES,
                value=WHISPER_MODEL_SIZE,
                label="Whisper model size (speed vs. accuracy)",
                info="tiny = fastest (default), small = most accurate. First use of a size loads it (one-time delay); cached after that.",
            )
            run_btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            english_out = gr.Textbox(label="English (transcribed)", lines=3)
            burmese_out = gr.Textbox(label="Burmese (translated)", lines=3)
            audio_out = gr.Audio(label="Burmese speech (synthesized)", type="numpy")
            log_out = gr.Textbox(label="Log", lines=5)

    run_btn.click(
        fn=run_pipeline,
        inputs=[mic_input, file_input, whisper_size_dd],
        outputs=[english_out, burmese_out, audio_out, log_out],
    )

if __name__ == "__main__":
    demo.queue().launch()