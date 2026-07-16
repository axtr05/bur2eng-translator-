# Offline Real-Time Speech Translation System

An end-to-end, completely offline translation application capable of real-time speech recognition, machine translation, and speech synthesis. It is designed for local classroom or group communication without requiring any cloud APIs.

---

## Features

- Offline English ↔ Burmese translation
- Live microphone translation
- Audio file translation
- Text translation
- Host–Client architecture
- Live subtitles
- Live translated speech

---

## Project Structure

```text
.
├── app.py
├── server.py
├── pipeline_manager.py
├── e2b.py
├── b2e.py
├── models/
├── static/
├── templates/
├── requirements.txt
└── README.md
```

---

## Requirements

- Windows 10/11
- Python 3.11+
- Git
- NVIDIA GPU (recommended)

---

## Installation

Clone the repository:
```bash
git clone https://github.com/axtr05/b_e_translate.git
cd b_e_translate
```

Create a virtual environment:
```bash
python -m venv venv
```

Activate the virtual environment:
```cmd
# Windows Command Prompt
venv\Scripts\activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

Generate HTTPS certificates (ONE TIME ONLY):
```bash
# Git Bash / Linux / macOS
./setup_mkcert.sh
```

---

## Download Whisper Small Myanmar

The custom ASR model must be downloaded from:
[https://huggingface.co/axtr05/whisper-small-myanmar](https://huggingface.co/axtr05/whisper-small-myanmar)

### Method 1: Using Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
hf download axtr05/whisper-small-myanmar --local-dir models/whisper-small-myanmar
```

### Method 2: Manual Download

Download all repository files from Hugging Face and extract them exactly here:
```text
models/
└── whisper-small-myanmar/
```

 the model is stored in b2e.py, update the `WHISPER_MODEL_PATH` variable.
 if you face any issues ask an ai model coz i can't answer any lol :>

---

## Run

Start the server:
```bash
python app.py
```

Access the application from your browser:
```text
Host   : https://<HOST-IP>:8000
Client : https://<HOST-IP>:8000/client
```

---

## Usage

1. Open the Host URL.
2. Open the Client URL on another device.
3. Select the translation direction on the Host device.
4. Start Live streaming or upload Audio/Text.
5. The Client receives translated subtitles and synthesized audio.

---

## Troubleshooting

- **Whisper model missing:** Ensure the model is located exactly in `models/whisper-small-myanmar`.
- **CUDA library not found:** Update your NVIDIA drivers and ensure they match your PyTorch CUDA version.
- **HTTPS certificate warning:** Click "Advanced" -> "Proceed", or install the local mkcert CA.
- **mkcert setup fails on Windows:** Ensure you have installed mkcert (`choco install mkcert`) or downloaded the binary.
- **Firewall blocking LAN:** Allow Python and port 8000 through your Windows Firewall.
- **Port already in use:** Change the port in `config.json`.
