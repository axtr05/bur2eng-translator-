import socket
import json
import os
import sys
from server import app, socketio

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "https_enabled": False,
            "port": 8000,
            "host": "0.0.0.0"
        }

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a public IP to determine the primary interface being used
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    config = load_config()
    host_ip = get_host_ip()
    port = config.get("port", 8000)
    host = config.get("host", "0.0.0.0")
    
    https_enabled = config.get("https_enabled", False)
    protocol = "https" if https_enabled else "http"
    
    print("\n" + "="*36)
    if https_enabled:
        print("HTTPS Enabled")
    print("Host Interface")
    print(f"{protocol}://{host_ip}:{port}")
    print("\nClient Interface")
    print(f"{protocol}://{host_ip}:{port}/client")
    print("="*36 + "\n")
    
    ssl_context = None
    if https_enabled:
        cert_path = config.get("cert_path", "certs/server.crt")
        key_path = config.get("key_path", "certs/server.key")
        
        if not os.path.exists(cert_path) or not os.path.exists(key_path):
            print("Certificates not found.")
            print("Run ./setup_mkcert.sh to generate them.")
            sys.exit(1)
            
        ssl_context = (cert_path, key_path)
    
    # Start the Flask-SocketIO server
    socketio.run(app, host=host, port=port, debug=True, use_reloader=False, ssl_context=ssl_context)
