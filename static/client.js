const socket = io({ 
    query: { type: "client" }, 
    secure: window.location.protocol === 'https:',
    transports: ["websocket", "polling"] 
});

// DOM Elements
const connectionIndicator = document.getElementById('connection-indicator');
const connectionText = document.getElementById('connection-text');
const languageIndicator = document.getElementById('language-indicator');
const currentStateText = document.getElementById('current-state');
const originalTranscript = document.getElementById('original-transcript');
const translatedTranscript = document.getElementById('translated-transcript');
const btnStart = document.getElementById('btn-start');
const modeSelect = document.getElementById('mode');

// Input Sections
const inputMicrophone = document.getElementById('input-microphone');
const inputUpload = document.getElementById('input-upload');
const inputText = document.getElementById('input-text');
const textInput = document.getElementById('text-input');
const fileInput = document.getElementById('audio-file');

// Audio Elements
const audioPlayer = document.getElementById('client-audio-player');
const audioIcon = document.getElementById('audio-icon');
const audioText = document.getElementById('audio-text');
const overlay = document.getElementById('autoplay-overlay');
const btnEnableAudio = document.getElementById('btn-enable-audio');

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

let isLocked = false;
let lockRole = null;
let audioEnabled = false;

// Audio Enable
btnEnableAudio.addEventListener('click', () => {
    audioEnabled = true;
    overlay.style.display = 'none';
    if (audioPlayer.src) {
        audioPlayer.play().catch(e => console.warn("Still blocked", e));
    }
});

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

function updateState(state) {
    if (isLocked && lockRole !== 'client') {
        currentStateText.textContent = '● Teacher Speaking... Please Wait';
        connectionIndicator.style.backgroundColor = 'var(--danger)';
    } else if (state === 'Idle' || state === 'Stopped' || state === 'Finished') {
        currentStateText.textContent = '● Idle';
        connectionIndicator.style.backgroundColor = 'var(--text-secondary)';
    } else if (state === 'Listening' || state === 'Broadcasting') {
        currentStateText.textContent = '● Listening';
        connectionIndicator.style.backgroundColor = 'var(--success)';
    } else {
        currentStateText.textContent = '● Processing';
        connectionIndicator.style.backgroundColor = 'var(--warning)';
    }
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
});

// Socket Events
socket.on('connect', () => {
    connectionIndicator.classList.add('connected');
    connectionText.textContent = 'Connected';
    
    // Log WebSocket upgrade
    const transport = socket.io.engine.transport.name;
    if (transport === 'websocket') {
        console.log('Transport Polling -> WebSocket Upgrade Successful');
    } else {
        socket.io.engine.on('upgrade', () => {
            if (socket.io.engine.transport.name === 'websocket') {
                console.log('Transport Polling -> WebSocket Upgrade Successful');
            }
        });
    }
});

socket.on('disconnect', () => {
    connectionIndicator.classList.remove('connected');
    connectionText.textContent = 'Disconnected';
});

socket.on('update_direction', (data) => {
    if (data.direction === 'e2b') {
        languageIndicator.textContent = 'English → Burmese';
    } else {
        languageIndicator.textContent = 'Burmese → English';
    }
});

socket.on('conversation_lock', (data) => {
    isLocked = true;
    lockRole = data.role;
    updateState('Locked');
    
    // Disable inputs if we are not the one who locked it
    if (data.role !== 'client') {
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

socket.on('ui_state', (data) => {
    updateState(data.state);
});

socket.on('timeline_update', (data) => {
    updateTimeline(data.stage);
});

socket.on('clear_client', () => {
    if (translatedTranscript.innerHTML.trim() !== "") {
        const separator = document.createElement('hr');
        separator.style.borderColor = "var(--border)";
        separator.style.margin = "2rem 0";
        translatedTranscript.appendChild(separator);
        originalTranscript.appendChild(separator.cloneNode());
    }
});

socket.on('translation_result', (data) => {
    // Both sides see original and translated transcripts appended
    if (data.original_transcript) {
        if (originalTranscript.textContent.includes("Processing...") || originalTranscript.textContent.includes("Awaiting input...") || originalTranscript.textContent.includes("Listening...")) {
            originalTranscript.innerHTML = "";
        }
        
        const origSpan = document.createElement('span');
        origSpan.textContent = data.original_transcript + " ";
        if (data.source === 'host') origSpan.style.color = "var(--text-primary)";
        if (data.source === 'client') origSpan.style.color = "var(--accent-color)";
        
        originalTranscript.appendChild(origSpan);
        originalTranscript.scrollTop = originalTranscript.scrollHeight;
    }

    if (data.translated_transcript) {
        if (translatedTranscript.textContent.includes("Processing...") || translatedTranscript.textContent.includes("Awaiting translation...") || translatedTranscript.textContent.includes("Waiting for speech...")) {
            translatedTranscript.innerHTML = "";
        }
        
        const transSpan = document.createElement('span');
        transSpan.textContent = data.translated_transcript + " ";
        if (data.source === 'host') transSpan.style.color = "var(--text-primary)";
        if (data.source === 'client') transSpan.style.color = "var(--accent-color)";
        
        translatedTranscript.appendChild(transSpan);
        translatedTranscript.scrollTop = translatedTranscript.scrollHeight;
    }
    
    // Only play audio if the source was someone else (the Host)
    if (data.audio_url && data.source !== 'client') {
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
            if (!audioEnabled) {
                overlay.style.display = 'flex';
            }
        });
    }
});

// Audio Playback UI
audioPlayer.addEventListener('play', () => {
    audioIcon.textContent = "🔊";
    audioIcon.classList.add('playing');
    audioText.textContent = "● Playing Audio";
});

audioPlayer.addEventListener('ended', () => {
    audioIcon.textContent = "⏸";
    audioIcon.classList.remove('playing');
    audioText.textContent = "● Waiting";
});

audioPlayer.addEventListener('pause', () => {
    if(audioPlayer.currentTime !== audioPlayer.duration && !audioPlayer.ended) {
        audioIcon.textContent = "⏸";
        audioIcon.classList.remove('playing');
        audioText.textContent = "● Waiting";
    }
});


// Web Audio API for Live Streaming
let audioContext;
let scriptProcessor;
let mediaStreamSource;
let stream;
let isStreaming = false;

async function startLiveStreaming() {
    if (isLocked && lockRole !== 'client') return;
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        if (audioContext.state === 'suspended') await audioContext.resume();
        
        mediaStreamSource = audioContext.createMediaStreamSource(stream);
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
            socket.emit('audio_stream', buffer);
        };
        
        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        
        isStreaming = true;
        updateState('Listening');
        socket.emit('start_live', { source: 'client' });
        
    } catch (e) {
        console.error("Microphone Access Failed:", e);
        btnStart.textContent = "🟢 Start Live";
        btnStart.style.backgroundColor = "var(--accent-color)";
        isStreaming = false;
    }
}

function stopLiveStreaming() {
    isStreaming = false;
    if (scriptProcessor) scriptProcessor.disconnect();
    if (mediaStreamSource) mediaStreamSource.disconnect();
    if (audioContext) audioContext.close();
    if (stream) stream.getTracks().forEach(track => track.stop());
    socket.emit('stop_live');
}

// UI Controls
btnStart.addEventListener('click', async () => {
    if (isLocked && lockRole !== 'client') return;
    
    const mode = modeSelect.value;
    
    if (mode === 'text') {
        originalTranscript.textContent = "Processing...";
        translatedTranscript.textContent = "Processing...";
        resetTimeline();
        
        const textPayload = textInput.value;
        if (!textPayload.trim()) return alert("Please enter text.");
        updateTimeline('stage-received');
        socket.emit('start_translation', { source: 'client', mode, text: textPayload });
        
    } else if (mode === 'upload') {
        originalTranscript.textContent = "Processing...";
        translatedTranscript.textContent = "Processing...";
        resetTimeline();
        
        if (fileInput.files.length === 0) return alert("Please select an audio file.");
        const file = fileInput.files[0];
        const arrayBuffer = await file.arrayBuffer();
        updateTimeline('stage-received');
        socket.emit('start_translation', { source: 'client', mode, audio: arrayBuffer });
        
    } else if (mode === 'microphone') {
        if (!isStreaming) {
            btnStart.textContent = "🔴 Stop Live";
            btnStart.style.backgroundColor = "var(--danger)";
            originalTranscript.textContent = "Listening...";
            translatedTranscript.textContent = "Waiting for speech...";
            resetTimeline();
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
