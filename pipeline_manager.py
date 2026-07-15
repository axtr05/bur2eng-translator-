import os
import time
import io
import traceback
import threading
import queue
import numpy as np
import soundfile as sf
import scipy.io.wavfile
import torch

# Suppress Transformers "max_length" warning (Issue 6)
import transformers
transformers.logging.set_verbosity_error()

import e2b
import b2e

# Ensure temp_audio directory exists
TEMP_DIR = 'temp_audio'
os.makedirs(TEMP_DIR, exist_ok=True)

def is_cuda_available():
    return torch.cuda.is_available()

def load_audio_from_bytes(audio_bytes):
    with io.BytesIO(audio_bytes) as b:
        data, samplerate = sf.read(b)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    return data, samplerate

# ==============================================================================
# TEMP AUDIO CLEANUP
# ==============================================================================
def cleanup_temp_files():
    """Background thread to delete .wav files older than 5 minutes."""
    while True:
        try:
            now = time.time()
            for filename in os.listdir(TEMP_DIR):
                filepath = os.path.join(TEMP_DIR, filename)
                if os.path.isfile(filepath) and filepath.endswith('.wav'):
                    # Delete if older than 300 seconds (5 mins)
                    if os.stat(filepath).st_mtime < now - 300:
                        os.remove(filepath)
        except Exception:
            pass
        time.sleep(60)

threading.Thread(target=cleanup_temp_files, daemon=True).start()

# ==============================================================================
# SINGLE-SHOT PIPELINE (Text & Upload)
# ==============================================================================
def process_translation(direction, mode, input_data, emit_func):
    total_start = time.time()
    result = {
        "original_transcript": "",
        "translated_transcript": "",
        "audio_url": "",
        "latency_sec": 0.0,
        "success": False
    }

    try:
        pipeline = e2b if direction == 'e2b' else b2e
        whisper_size = pipeline.WHISPER_MODEL_SIZE
            
        emit_func('status', 'Transcribing' if mode != 'text' else 'Translating')

        audio_input = None
        orig_text = ""

        if mode == 'text':
            orig_text = input_data
        else:
            emit_func('log', 'Decoding audio file...')
            raw_audio, orig_sr = load_audio_from_bytes(input_data)
            audio = pipeline.resample_if_needed(raw_audio, orig_sr, pipeline.SAMPLE_RATE)
            
            emit_func('log', 'Running voice activity detection...')
            speech_audio, vad_err = pipeline.vad_filter(audio)
            if vad_err:
                emit_func('log', vad_err)
                raise Exception("VAD Filter failed or audio too silent.")
            
            emit_func('log', 'Running ASR')
            asr_start = time.time()
            if direction == 'e2b':
                orig_text = pipeline.transcribe_english(speech_audio, whisper_size)
            else:
                orig_text = pipeline.transcribe_burmese(speech_audio, whisper_size)
            
            if not orig_text.strip():
                raise Exception("Empty transcription (ASR stage failed)")

        result["original_transcript"] = orig_text
        emit_func('timeline', 'stage-asr')
        
        emit_func('status', 'Translating')
        emit_func('log', 'Running Translation')
        
        if direction == 'e2b':
            translated_text = pipeline.translate_to_burmese(orig_text)
        else:
            translated_text = pipeline.translate_to_english(orig_text)
            
        if not translated_text.strip():
            raise Exception("Empty translation (Translation stage failed)")
            
        result["translated_transcript"] = translated_text
        emit_func('timeline', 'stage-mt')
        
        emit_func('status', 'Generating Speech')
        emit_func('log', 'Generating Speech')
        
        if direction == 'e2b':
            sr, waveform = pipeline.speak_burmese(translated_text)
        else:
            sr, waveform = pipeline.speak_english(translated_text)
            
        emit_func('timeline', 'stage-tts')
        
        output_filename = f"output_{int(time.time())}.wav"
        output_path = os.path.join(TEMP_DIR, output_filename)
        scipy.io.wavfile.write(output_path, sr, waveform)
        
        result["audio_url"] = f"/{TEMP_DIR}/{output_filename}"
        
        emit_func('status', 'Broadcasting')
        emit_func('log', 'Broadcast Complete')
        emit_func('timeline', 'stage-broadcast')
        
        total_time = time.time() - total_start
        result["latency_sec"] = round(total_time, 2)
        result["success"] = True

    except Exception as e:
        emit_func('log', f"Translation Failed: {str(e)}")
        emit_func('status', 'Finished')
        traceback.print_exc()
        result["success"] = False
        
    return result

# ==============================================================================
# LIVE CONTINUOUS STREAMING (Producer-Consumer)
# ==============================================================================

live_active = False
raw_audio_queue = queue.Queue()
asr_queue = queue.Queue()
mt_queue = queue.Queue()
tts_queue = queue.Queue()
broadcast_queue = queue.Queue()

workers = []
global_emit = None

def push_live_audio(buffer):
    if live_active:
        msg = f"Chunk Received size {len(buffer)} timestamp {time.time():.2f}"
        print(msg)
        if global_emit:
            global_emit('log', msg)
        raw_audio_queue.put(buffer)

def vad_worker(direction, emit_func):
    """Consumes raw float32 streams, detects speech using Silero VAD, and chunks it."""
    window_size = 512
    buffer = np.array([], dtype=np.float32)
    speech_buffer = []
    is_speaking = False
    silence_counter = 0
    speech_frames = 0
    # 0.7 sec of silence indicates the end of a speech segment
    SILENCE_THRESHOLD = int(16000 * 0.7) 
    
    while live_active:
        try:
            chunk = raw_audio_queue.get(timeout=0.5)
            if chunk is None: break
            
            data = np.frombuffer(chunk, dtype=np.float32)
            buffer = np.concatenate((buffer, data))
            
            while len(buffer) >= window_size:
                window = buffer[:window_size]
                buffer = buffer[window_size:]
                
                tensor = torch.from_numpy(window)
                prob = e2b.vad_model(tensor, 16000).item()
                
                if prob > 0.5:
                    if not is_speaking:
                        emit_func('timeline', 'stage-received')
                    is_speaking = True
                    silence_counter = 0
                    speech_frames += 1
                    speech_buffer.append(window)
                else:
                    if is_speaking:
                        silence_counter += window_size
                        speech_buffer.append(window)
                        if silence_counter >= SILENCE_THRESHOLD:
                            emit_func('timeline', 'stage-vad')
                            segment = np.concatenate(speech_buffer)
                            
                            dur = len(segment) / 16000.0
                            true_speech_dur = speech_frames * (window_size / 16000.0)
                            
                            # Ignore tiny noises (e.g. keyboard clicks) if actual speech is < 150ms
                            if true_speech_dur >= 0.15:
                                msg = f"Speech Segment Created duration {dur:.2f}s"
                                print(msg)
                                emit_func('log', msg)
                                asr_queue.put({'audio': segment, 'direction': direction})
                            else:
                                emit_func('log', f"Ignored short noise segment (true speech: {true_speech_dur:.2f}s)")
                            
                            speech_buffer = []
                            is_speaking = False
                            silence_counter = 0
                            speech_frames = 0
        except queue.Empty:
            continue
        except Exception as e:
            emit_func('log', f"VAD Error: {str(e)}")

def asr_worker(emit_func):
    """Consumes audio segments and transcribes them."""
    while live_active:
        try:
            item = asr_queue.get(timeout=0.5)
            if item is None: break
            
            direction = item['direction']
            audio = item['audio']
            pipeline = e2b if direction == 'e2b' else b2e
            
            emit_func('timeline', 'stage-asr')
            emit_func('status', 'Transcribing')
            
            msg = "ASR Started"
            print(msg)
            emit_func('log', msg)
            
            start_t = time.time()
            if direction == 'e2b':
                text = pipeline.transcribe_english(audio, pipeline.WHISPER_MODEL_SIZE)
            else:
                text = pipeline.transcribe_burmese(audio, pipeline.WHISPER_MODEL_SIZE)
            
            msg_finish = "ASR Finished"
            print(msg_finish)
            emit_func('log', msg_finish)
            
            asr_time = time.time() - start_t
            
            if text.strip():
                # Emit LIVE original transcript immediately for the Host UI
                emit_func('result', {
                    "original_transcript": text,
                    "translated_transcript": "",
                    "audio_url": ""
                })
                mt_queue.put({'text': text, 'direction': direction, 'asr_time': asr_time})
            else:
                emit_func('log', "Translation Failed: ASR returned empty transcript")
                
        except queue.Empty:
            continue
        except Exception as e:
            emit_func('log', f"Translation Failed (ASR): {str(e)}")

def mt_worker(emit_func):
    """Consumes transcripts and translates them."""
    while live_active:
        try:
            item = mt_queue.get(timeout=0.5)
            if item is None: break
            
            direction = item['direction']
            orig_text = item['text']
            pipeline = e2b if direction == 'e2b' else b2e
            
            emit_func('timeline', 'stage-mt')
            emit_func('status', 'Translating')
            
            msg = "Running Translation"
            print(msg)
            emit_func('log', msg)
            
            start_t = time.time()
            if direction == 'e2b':
                translated_text = pipeline.translate_to_burmese(orig_text)
            else:
                translated_text = pipeline.translate_to_english(orig_text)
                
            mt_time = time.time() - start_t
            
            if translated_text.strip():
                tts_queue.put({
                    'orig_text': orig_text,
                    'trans_text': translated_text,
                    'direction': direction,
                    'asr_time': item['asr_time'],
                    'mt_time': mt_time
                })
            else:
                emit_func('log', "Translation Failed: MT returned empty translation")
                
        except queue.Empty:
            continue
        except Exception as e:
            emit_func('log', f"Translation Failed (MT): {str(e)}")

def tts_worker(emit_func):
    """Consumes translations and synthesizes speech."""
    while live_active:
        try:
            item = tts_queue.get(timeout=0.5)
            if item is None: break
            
            direction = item['direction']
            trans_text = item['trans_text']
            pipeline = e2b if direction == 'e2b' else b2e
            
            emit_func('timeline', 'stage-tts')
            emit_func('status', 'Generating Speech')
            
            msg = "Generating Speech"
            print(msg)
            emit_func('log', msg)
            
            start_t = time.time()
            if direction == 'e2b':
                sr, waveform = pipeline.speak_burmese(trans_text)
            else:
                sr, waveform = pipeline.speak_english(trans_text)
                
            tts_time = time.time() - start_t
            
            # Save audio
            output_filename = f"live_{int(time.time() * 1000)}.wav"
            output_path = os.path.join(TEMP_DIR, output_filename)
            scipy.io.wavfile.write(output_path, sr, waveform)
            
            broadcast_queue.put({
                'orig_text': item['orig_text'],
                'trans_text': trans_text,
                'audio_url': f"/{TEMP_DIR}/{output_filename}",
                'latency': round(item['asr_time'] + item['mt_time'] + tts_time, 2)
            })
            
        except queue.Empty:
            continue
        except Exception as e:
            emit_func('log', f"Translation Failed (TTS): {str(e)}")

def broadcast_worker(emit_func):
    """Consumes final payloads and broadcasts them."""
    while live_active:
        try:
            item = broadcast_queue.get(timeout=0.5)
            if item is None: break
            
            emit_func('timeline', 'stage-broadcast')
            emit_func('status', 'Broadcasting')
            
            msg = "Broadcast Complete"
            print(msg)
            emit_func('log', msg)
            
            emit_func('result', {
                "original_transcript": "", # Already emitted by ASR worker
                "translated_transcript": item['trans_text'],
                "audio_url": item['audio_url']
            })
            
            emit_func('latency', item['latency'])
            emit_func('status', 'Listening')
            
        except queue.Empty:
            continue

def start_live_pipeline(direction, emit_func):
    global live_active, workers, global_emit
    global_emit = emit_func
    
    while not raw_audio_queue.empty(): raw_audio_queue.get()
    while not asr_queue.empty(): asr_queue.get()
    while not mt_queue.empty(): mt_queue.get()
    while not tts_queue.empty(): tts_queue.get()
    while not broadcast_queue.empty(): broadcast_queue.get()
    
    live_active = True
    
    workers = [
        threading.Thread(target=vad_worker, args=(direction, emit_func), daemon=True),
        threading.Thread(target=asr_worker, args=(emit_func,), daemon=True),
        threading.Thread(target=mt_worker, args=(emit_func,), daemon=True),
        threading.Thread(target=tts_worker, args=(emit_func,), daemon=True),
        threading.Thread(target=broadcast_worker, args=(emit_func,), daemon=True)
    ]
    
    for w in workers:
        w.start()
        
    emit_func('status', 'Listening')

def stop_live_pipeline():
    global live_active, global_emit
    live_active = False
    global_emit = None
    
    raw_audio_queue.put(None)
    asr_queue.put(None)
    mt_queue.put(None)
    tts_queue.put(None)
    broadcast_queue.put(None)
