from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
import os

import pipeline_manager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'offline_speech_translation_secret'

# Increase buffer size for large audio arrays
# async_mode='threading' to avoid "Invalid frame header" conflicts with default Eventlet/Gevent fallbacks
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*", max_http_buffer_size=50 * 1024 * 1024)

# Create temp_audio directory
os.makedirs('temp_audio', exist_ok=True)

active_client_sids = set()
active_host_sids = set()
global_direction = 'e2b'
current_speaker_sid = None
current_speaker_role = None

def acquire_lock(sid, role):
    global current_speaker_sid, current_speaker_role
    # If it's unlocked, or if the same role is trying to acquire it again (e.g. Host refreshed)
    if current_speaker_role is None or current_speaker_role == role:
        current_speaker_sid = sid
        current_speaker_role = role
        socketio.emit('conversation_lock', {'role': role, 'sid': sid})
        return True
    return False

def release_lock(role):
    global current_speaker_sid, current_speaker_role
    if current_speaker_role == role:
        current_speaker_sid = None
        current_speaker_role = None
        socketio.emit('conversation_unlock')

# ==============================================================================
# HTTP ROUTES
# ==============================================================================

@app.route('/')
def host_page():
    return render_template('host.html')

@app.route('/client')
def client_page():
    return render_template('client.html')

@app.route('/temp_audio/<path:filename>')
def download_temp_audio(filename):
    return send_from_directory('temp_audio', filename)

# ==============================================================================
# SOCKET.IO EVENTS
# ==============================================================================

@socketio.on('connect')
def handle_connect():
    client_type = request.args.get('type')
    
    if client_type == 'client':
        active_client_sids.add(request.sid)
        emit('client_count_update', {'count': len(active_client_sids)}, broadcast=True)
        emit('system_log', {'message': f'New client connected. Total: {len(active_client_sids)}'}, broadcast=True)
        
        # Send current state to the new client
        emit('update_direction', {'direction': global_direction})
        if current_speaker_role is not None:
            emit('conversation_lock', {'role': current_speaker_role, 'sid': current_speaker_sid})
            
    elif client_type == 'host':
        active_host_sids.add(request.sid)
        # Send current state to the connecting host
        emit('client_count_update', {'count': len(active_client_sids)})
        emit('update_direction', {'direction': global_direction})
        if current_speaker_role is not None:
            emit('conversation_lock', {'role': current_speaker_role, 'sid': current_speaker_sid})
    
    gpu_type = 'CUDA' if pipeline_manager.is_cuda_available() else 'CPU'
    emit('gpu_status', {'gpu': gpu_type})

@socketio.on('disconnect')
def handle_disconnect():
    global current_speaker_sid, current_speaker_role, global_direction
    
    if request.sid in active_client_sids:
        active_client_sids.remove(request.sid)
        emit('client_count_update', {'count': len(active_client_sids)}, broadcast=True)
        emit('system_log', {'message': f'Client disconnected. Total: {len(active_client_sids)}'}, broadcast=True)
        
        # If this specific client was speaking, release the lock
        if current_speaker_sid == request.sid:
            release_lock('client')
            
    elif request.sid in active_host_sids:
        active_host_sids.remove(request.sid)
        emit('system_log', {'message': f'Host disconnected.'}, broadcast=True)
        
        # Only reset global state if no clients remain connected
        if len(active_client_sids) == 0:
            current_speaker_sid = None
            current_speaker_role = None
            global_direction = 'e2b'
            emit('conversation_unlock', broadcast=True)

@socketio.on('clear_client_session')
def handle_clear_client():
    emit('clear_client', broadcast=True)

@socketio.on('change_direction')
def handle_change_direction(data):
    global global_direction
    global_direction = data.get('direction', 'e2b')
    emit('update_direction', data, broadcast=True)

# ------------------------------------------------------------------------------
# Single-Shot Processing (Text & Upload)
# ------------------------------------------------------------------------------
@socketio.on('start_translation')
def handle_start_translation(data):
    source = data.get('source', 'host')
    mode = data.get('mode', 'text')
    input_data = data.get('audio') if mode != 'text' else data.get('text', '')
    
    if not acquire_lock(request.sid, source):
        emit('system_log', {'message': 'Error: Conversation is currently locked.'})
        return

    # Determine pipeline direction
    direction = global_direction
    if source == 'client':
        # Reverse direction for client
        direction = 'b2e' if global_direction == 'e2b' else 'e2b'
    
    if not input_data:
        emit('system_log', {'message': 'Error: No input data provided.'})
        emit('ui_state', {'state': 'Finished'})
        release_lock(source)
        return
        
    def emit_wrapper(event_type, msg):
        if event_type == 'log':
            emit('system_log', {'message': msg})
        elif event_type == 'status':
            emit('ui_state', {'state': msg})
        elif event_type == 'timeline':
            emit('timeline_update', {'stage': msg})
            
    emit_wrapper('status', 'Loading Models' if not hasattr(pipeline_manager, 'models_ready') else 'Ready')
    pipeline_manager.models_ready = True
    
    result = pipeline_manager.process_translation(direction, mode, input_data, emit_wrapper)
    
    # Inject source so frontend knows who spoke
    result['source'] = source
    
    
    if result["success"]:
        emit('latency_update', {'latency': result["latency_sec"]})
        emit('translation_result', result, broadcast=True)
    
    emit_wrapper('status', 'Finished')
    release_lock(source)

@socketio.on('stop_translation')
def handle_stop_translation():
    client_type = 'host' if request.sid in active_host_sids else 'client'
    release_lock(client_type)
    emit('system_log', {'message': 'Translation pipeline stopped manually.'}, broadcast=True)
    emit('ui_state', {'state': 'Stopped'})

# ------------------------------------------------------------------------------
# Live Continuous Streaming (Microphone)
# ------------------------------------------------------------------------------
@socketio.on('start_live')
def handle_start_live(data):
    source = data.get('source', 'host')
    
    if not acquire_lock(request.sid, source):
        emit('system_log', {'message': 'Error: Conversation is currently locked.'})
        return
        
    # Reverse direction for client
    direction = global_direction
    if source == 'client':
        direction = 'b2e' if global_direction == 'e2b' else 'e2b'
    
    def emit_wrapper(event_type, msg):
        if event_type == 'log':
            socketio.emit('system_log', {'message': msg})
        elif event_type == 'status':
            socketio.emit('ui_state', {'state': msg})
        elif event_type == 'timeline':
            socketio.emit('timeline_update', {'stage': msg})
        elif event_type == 'latency':
            socketio.emit('latency_update', {'latency': msg})
        elif event_type == 'result':
            # msg is the dict containing transcripts
            msg['source'] = source
            socketio.emit('translation_result', msg)

    print(f"Live Started by {source}")
    socketio.emit('system_log', {'message': f'Starting Live Stream [{direction}]...'})
    pipeline_manager.start_live_pipeline(direction, emit_wrapper)

@socketio.on('audio_stream')
def handle_audio_stream(audio_buffer):
    # Determine the role of the sender
    client_type = 'host' if request.sid in active_host_sids else 'client'
    # Only allow the current lock holder role to stream audio
    if current_speaker_role == client_type:
        pipeline_manager.push_live_audio(audio_buffer)

@socketio.on('stop_live')
def handle_stop_live():
    client_type = 'host' if request.sid in active_host_sids else 'client'
    if current_speaker_role == client_type:
        pipeline_manager.stop_live_pipeline()
        release_lock(client_type)
        socketio.emit('system_log', {'message': 'Live Stream stopped.'})
        socketio.emit('ui_state', {'state': 'Stopped'})
