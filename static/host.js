const socket = io({ 
    query: { type: "host" }, 
    secure: window.location.protocol === 'https:',
    transports: ["websocket", "polling"] 
});

// DOM Elements
const connectionIndicator = document.getElementById('connection-indicator');
const clientCount = document.getElementById('client-count');
const logs = document.getElementById('logs');
const originalTranscript = document.getElementById('original-transcript');
const translatedTranscript = document.getElementById('translated-transcript');
const audioPlayer = document.getElementById('audio-player');
const btnStart = document.getElementById('btn-start');
const directionSelect = document.getElementById('direction');
const modeSelect = document.getElementById('mode');

// Input Sections
const inputMicrophone = document.getElementById('input-microphone');
const inputUpload = document.getElementById('input-upload');
const inputText = document.getElementById('input-text');
const textInput = document.getElementById('text-input');
const fileInput = document.getElementById('audio-file');

// Autoplay Overlay
const overlay = document.getElementById('autoplay-overlay');
const btnEnableAudio = document.getElementById('btn-enable-audio');

let isLocked = false;
let lockRole = null;
let audioEnabled = false;

if (btnEnableAudio) {
    btnEnableAudio.addEventListener('click', () => {
        audioEnabled = true;
        overlay.style.display = 'none';
        if (audioPlayer.src) {
            audioPlayer.play().catch(e => console.warn("Still blocked", e));
        }
    });
}

// Status Bar Elements
const stateIndicator = document.getElementById('connection-indicator'); 
const currentStateText = document.getElementById('current-state');
const currentPipelineText = document.getElementById('current-pipeline');
const gpuStatus = document.getElementById('gpu-status');
const latencyVal = document.getElementById('latency-val');

// Timeline Elements
const timelineStages = {
    'stage-received': document.getElementById('stage-received'),
    'stage-vad': document.getElementById('stage-vad'),
    'stage-asr': document.getElementById('stage-asr'),
    'stage-mt': document.getElementById('stage-mt'),
    'stage-tts': document.getElementById('stage-tts'),
    'stage-broadcast': document.getElementById('stage-broadcast')
};

const stageOrder = ['stage-received', 'stage-vad', 'stage-asr', 'stage-mt', 'stage-tts', 'stage-broadcast'];

function resetTimeline() {
    stageOrder.forEach(id => {
        timelineStages[id].classList.remove('active', 'complete');
    });
}

function updateTimeline(activeStageId) {
    let passed = true;
    stageOrder.forEach(id => {
        const el = timelineStages[id];
        el.classList.remove('active', 'complete');
        if (id === activeStageId) {
            el.classList.add('active');
            passed = false;
        } else if (passed) {
            el.classList.add('complete');
        }
    });
}

// Handle Input Mode Switching
modeSelect.addEventListener('change', () => {
    const mode = modeSelect.value;
    inputMicrophone.classList.remove('active');
    inputUpload.classList.remove('active');
    inputText.classList.remove('active');
    
    btnStart.textContent = mode === 'microphone' ? '🟢 Start Live' : 'Translate';
    btnStart.style.backgroundColor = "var(--accent-color)";
    
    if (mode === 'microphone') inputMicrophone.classList.add('active');
    if (mode === 'upload') inputUpload.classList.add('active');
    if (mode === 'text') inputText.classList.add('active');
    
    socket.emit('clear_client_session');
});

function updateState(state) {
    if (isLocked && lockRole === 'client') {
        currentStateText.textContent = '● Student Speaking... Please Wait';
        stateIndicator.style.backgroundColor = 'var(--danger)';
    } else if (state === 'Idle' || state === 'Stopped' || state === 'Finished') {
        currentStateText.textContent = '● Idle';
        stateIndicator.style.backgroundColor = 'var(--text-secondary)';
    } else if (state === 'Listening' || state === 'Broadcasting') {
        currentStateText.textContent = '● Listening';
        stateIndicator.style.backgroundColor = 'var(--success)';
    } else {
        currentStateText.textContent = '● Processing';
        stateIndicator.style.backgroundColor = 'var(--warning)';
    }
}

// Update pipeline text based on dropdown
directionSelect.addEventListener('change', () => {
    const dir = directionSelect.value;
    currentPipelineText.textContent = dir === 'e2b' ? 'English → Burmese' : 'Burmese → English';
    socket.emit('change_direction', { direction: dir });
    socket.emit('clear_client_session');
});

// Helper to add logs
function addLog(message) {
    const logEntry = document.createElement('div');
    const time = new Date().toLocaleTimeString();
    logEntry.textContent = `[${time}] ${message}`;
    logs.appendChild(logEntry);
    
    const logsContainer = document.querySelector('.logs-container');
    if (logsContainer.open) {
        logs.scrollTop = logs.scrollHeight;
    }
}

// Socket Events
socket.on('connect', () => {
    connectionIndicator.classList.add('connected');
    addLog('Connected to backend server.');
    
    // Log WebSocket upgrade
    const transport = socket.io.engine.transport.name;
    if (transport === 'websocket') {
        addLog('Transport Polling -> WebSocket Upgrade Successful');
    } else {
        socket.io.engine.on('upgrade', () => {
            if (socket.io.engine.transport.name === 'websocket') {
                addLog('Transport Polling -> WebSocket Upgrade Successful');
            }
        });
    }
    
    // Removed change_direction on connect to prevent state reset on refresh
});

socket.on('disconnect', () => {
    connectionIndicator.classList.remove('connected');
    addLog('Disconnected from backend server.');
});

socket.on('client_count_update', (data) => {
    clientCount.textContent = data.count;
});

socket.on('gpu_status', (data) => {
    gpuStatus.textContent = data.gpu;
    if (data.gpu === 'CPU') {
        gpuStatus.style.color = 'var(--warning)';
    }
});

socket.on('ui_state', (data) => {
    updateState(data.state);
});

socket.on('conversation_lock', (data) => {
    isLocked = true;
    lockRole = data.role;
    updateState('Locked');
    
    // Disable inputs if we are not the one who locked it
    if (data.role !== 'host') {
        btnStart.disabled = true;
        btnStart.style.opacity = '0.5';
    }
});

socket.on('conversation_unlock', () => {
    isLocked = false;
    lockRole = null;
    updateState('Idle');
    btnStart.disabled = false;
    btnStart.style.opacity = '1.0';
});

socket.on('timeline_update', (data) => {
    updateTimeline(data.stage);
});

socket.on('latency_update', (data) => {
    latencyVal.textContent = data.latency + ' sec';
});

socket.on('system_log', (data) => {
    addLog(data.message);
});

socket.on('translation_result', (data) => {
    if (data.original_transcript) {
        if (originalTranscript.textContent === "Processing..." || originalTranscript.textContent === "Awaiting input..." || originalTranscript.textContent === "Listening...") {
            originalTranscript.textContent = "";
        }
        
        const origSpan = document.createElement('span');
        origSpan.textContent = data.original_transcript + " ";
        if (data.source === 'host') origSpan.style.color = "var(--text-primary)";
        if (data.source === 'client') origSpan.style.color = "var(--accent-color)";
        
        originalTranscript.appendChild(origSpan);
        originalTranscript.scrollTop = originalTranscript.scrollHeight;
    }

    if (data.translated_transcript) {
        if (translatedTranscript.textContent === "Processing..." || translatedTranscript.textContent === "Awaiting translation..." || translatedTranscript.textContent === "Waiting for speech...") {
            translatedTranscript.textContent = "";
        }
        
        const transSpan = document.createElement('span');
        transSpan.textContent = data.translated_transcript + " ";
        if (data.source === 'host') transSpan.style.color = "var(--text-primary)";
        if (data.source === 'client') transSpan.style.color = "var(--accent-color)";
        
        translatedTranscript.appendChild(transSpan);
        translatedTranscript.scrollTop = translatedTranscript.scrollHeight;
    }
    
    // Host only plays audio if it came from the Client
    if (data.audio_url && data.source !== 'host') {
        if (!audioPlayer.paused) {
            audioPlayer.pause();
        }
        audioPlayer.removeAttribute('src');
        audioPlayer.load();
        
        audioPlayer.src = data.audio_url;
        
        audioPlayer.play().then(() => {
            audioEnabled = true;
        }).catch(e => {
            console.warn("Autoplay blocked:", e);
            if (!audioEnabled && overlay) {
                overlay.style.display = 'flex';
            }
        });
    }
});

// Web Audio API for Live Streaming
let audioContext;
let scriptProcessor;
let mediaStreamSource;
let stream;
let isStreaming = false;
let chunkCounter = 0;

async function startLiveStreaming() {
    console.log("Start Live Clicked");
    console.log("Requesting Microphone");
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("Permission Granted");
        
        // Enforce 16kHz sample rate for Silero VAD compatibility
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        console.log("AudioContext Created");
        console.log("AudioContext Running");
        
        mediaStreamSource = audioContext.createMediaStreamSource(stream);
        console.log("MediaStream Created");
        
        // 4096 buffer size is a safe middle ground for processing latency
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        scriptProcessor.onaudioprocess = (event) => {
            if (!isStreaming) return;
            
            // Backpressure: drop audio chunk if socket.io send buffer is backing up
            if (socket.sendBuffer && socket.sendBuffer.length > 2) {
                console.warn("Socket.IO send buffer backing up. Dropping audio chunk to preserve low latency.");
                return;
            }
            
            const float32Array = event.inputBuffer.getChannelData(0);
            const buffer = float32Array.slice().buffer;
            
            chunkCounter++;
            if (chunkCounter % 10 === 0) {
                // Print every ~2.5 seconds
                console.log(`Sending Audio Chunk #${chunkCounter}`);
            }
            
            socket.emit('audio_stream', buffer);
        };
        
        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        console.log("Audio Processor Started");
        
        isStreaming = true;
        addLog("Microphone Started");
        updateState('Listening');
        
        // Tell backend to spin up continuous workers
        socket.emit('start_live', { source: 'host', direction: directionSelect.value });
        
    } catch (e) {
        console.error("Microphone Access Failed:", e);
        addLog("Microphone access denied or error: " + e.message);
        btnStart.textContent = "🟢 Start Live";
        btnStart.style.backgroundColor = "var(--accent-color)";
        isStreaming = false;
    }
}

function stopLiveStreaming() {
    console.log("Stop Live Clicked");
    isStreaming = false;
    if (scriptProcessor) {
        scriptProcessor.disconnect();
    }
    if (mediaStreamSource) {
        mediaStreamSource.disconnect();
    }
    if (audioContext) {
        audioContext.close();
    }
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    addLog("Live Microphone streaming stopped.");
    socket.emit('stop_live');
}

// UI Controls
btnStart.addEventListener('click', async () => {
    if (isLocked && lockRole !== 'host') return;
    
    console.log("btnStart Clicked. Mode:", modeSelect.value);
    const direction = directionSelect.value;
    const mode = modeSelect.value;
    
    if (mode === 'text') {
        originalTranscript.textContent = "Processing...";
        translatedTranscript.textContent = "Processing...";
        resetTimeline();
        
        const textPayload = textInput.value;
        if (!textPayload.trim()) return alert("Please enter text.");
        updateTimeline('stage-received');
        socket.emit('start_translation', { source: 'host', direction, mode, text: textPayload });
        
    } else if (mode === 'upload') {
        originalTranscript.textContent = "Processing...";
        translatedTranscript.textContent = "Processing...";
        resetTimeline();
        
        if (fileInput.files.length === 0) return alert("Please select an audio file.");
        const file = fileInput.files[0];
        const arrayBuffer = await file.arrayBuffer();
        updateTimeline('stage-received');
        socket.emit('start_translation', { source: 'host', direction, mode, audio: arrayBuffer });
        
    } else if (mode === 'microphone') {
        if (!isStreaming) {
            btnStart.textContent = "🔴 Stop Live";
            btnStart.style.backgroundColor = "var(--danger)";
            originalTranscript.textContent = "Listening...";
            translatedTranscript.textContent = "Waiting for speech...";
            resetTimeline();
            socket.emit('clear_client_session');
            startLiveStreaming();
        } else {
            btnStart.textContent = "🟢 Start Live";
            btnStart.style.backgroundColor = "var(--accent-color)";
            stopLiveStreaming();
            updateState('Stopped');
            resetTimeline();
        }
    }
});

// Initialize button text on load
if (modeSelect.value === 'microphone') {
    btnStart.textContent = '🟢 Start Live';
}
