# Local HTTPS Setup with mkcert

To stream live microphone audio over the local network (LAN) without encountering browser "Not Secure" privacy warnings or blocked WebSockets, this project uses `mkcert`.

`mkcert` automatically generates locally-trusted development certificates.

## 1. Installation

Install `mkcert` on the Host machine:

### Ubuntu / Debian
```bash
sudo apt install libnss3-tools
brew install mkcert
# Or download the pre-built binary: https://github.com/FiloSottile/mkcert/releases
```

### Arch Linux
```bash
sudo pacman -S mkcert
```

### macOS
```bash
brew install mkcert
```

### Windows
```powershell
choco install mkcert
```

---

## 2. Certificate Generation

Once `mkcert` is installed on your Host machine, run the setup script:

```bash
chmod +x setup_mkcert.sh
./setup_mkcert.sh
```

This script will:
1. Run `mkcert -install` to install the local Root CA in your system trust store.
2. Auto-detect your LAN IP address (e.g. `192.168.1.100`) and Hostname.
3. Generate `certs/server.crt` and `certs/server.key` explicitly bound to `localhost`, `127.0.0.1`, `::1`, your LAN IP, and Hostname.

---

## 3. Running the Server

Start the application normally:

```bash
python app.py
```

The output will confirm HTTPS is active:

```text
HTTPS Enabled
Host Interface
https://<host-ip>:8000

Client Interface
https://<host-ip>:8000/client
```

When you visit `https://localhost:8000` on the Host machine, the browser will report a **Secure** connection! No privacy warnings will appear, and the microphone will be immediately accessible.

---

## 4. Connecting Client Devices on the LAN

If you try to connect another PC, laptop, or phone to `https://<host-ip>:8000/client`, they will still see a "Connection is not private" warning. This is because their browser does not know about the Host's custom `mkcert` Root CA.

To permanently fix this and make the connection trusted on the Client device:

### On the Host Machine
Find where your `rootCA.pem` file is stored:
```bash
mkcert -CAROOT
```
Copy the `rootCA.pem` file from that directory to the Client device (e.g., via USB drive, email, or a local file share).

### On the Client Device

**Windows / macOS / Linux:**
1. Install `mkcert` on the Client device (see Section 1).
2. Open a terminal on the Client and set the `CAROOT` variable to point to the folder containing the Host's `rootCA.pem`:
   ```bash
   export CAROOT=/path/to/folder/containing/rootCA.pem
   mkcert -install
   ```

**Android (Manual Installation without mkcert):**
1. Transfer `rootCA.pem` to your Android device.
2. Go to Settings > Security > Encryption & Credentials > Install a certificate > CA certificate.
3. Select the `rootCA.pem` file.

**iOS (Manual Installation without mkcert):**
1. AirDrop or email `rootCA.pem` to your iPhone.
2. Tap the file to download the profile.
3. Go to Settings > Profile Downloaded and install it.
4. Go to Settings > General > About > Certificate Trust Settings and toggle the switch for the mkcert Root CA.

Once the CA is installed, all Client devices on the LAN will connect securely to `https://<host-ip>:8000/client` with green padlocks and fully functioning websocket and microphone APIs.

---

## 5. Troubleshooting & Regeneration

If your Host machine's IP address changes (e.g. your router assigns you a new DHCP lease), the certificate will no longer match the IP.

To regenerate certificates for a new IP:
```bash
./setup_mkcert.sh
```
Restart `app.py` after regenerating. You do NOT need to reinstall the Root CA on client devices; the Root CA stays the same even if the server certificates are regenerated.
