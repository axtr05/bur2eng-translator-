# HTTPS Setup Guide

To use the Live Microphone features of this application (Host and Client), the browser's `navigator.mediaDevices.getUserMedia()` API is required. Modern browsers restrict this API to secure contexts (`https://` or `localhost`). 

Since you are hosting this application on your local network (LAN) and accessing it from other devices, you **must** serve it over HTTPS.

## Generating Development Certificates

The application includes an automated script to generate self-signed SSL certificates using OpenSSL.

To generate the certificates, run:
```bash
python generate_cert.py
```

This will create a `certs/` directory containing:
- `server.key` (Private Key)
- `server.crt` (Public Certificate)

## Configuration

Ensure that `config.json` is set to use HTTPS:
```json
{
    "https_enabled": true,
    "cert_path": "certs/server.crt",
    "key_path": "certs/server.key",
    "host": "0.0.0.0",
    "port": 8000
}
```

Once generated, you can simply run `python app.py` and the application will automatically start in HTTPS mode.

## Accessing from Another Device

When you open the application on your phone, tablet, or another laptop on the network, you will use the `https://` prefix.

1. Find the IP printed in the terminal (e.g. `https://192.168.1.100:8000/client`).
2. Type this exact URL into the browser on the Client device.

### Bypassing Browser Warnings

Because the script generates a *self-signed* certificate rather than one purchased from a trusted Certificate Authority (like Verisign), your browser will display a security warning (e.g., "Your connection is not private" or "Warning: Potential Security Risk Ahead").

**This is entirely normal and safe for a local development network.**

To proceed safely:
- **Chrome / Edge**: Click `Advanced`, then click `Proceed to [IP] (unsafe)`.
- **Firefox**: Click `Advanced`, then click `Accept the Risk and Continue`.
- **Safari**: Click `Show Details`, then click `visit this website`, and confirm with your device passcode if prompted.

Once you accept the risk, the browser will establish a secure HTTPS connection and grant access to the microphone API.
