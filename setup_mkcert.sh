#!/bin/bash

# Exit on any error
set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting mkcert setup for Local HTTPS...${NC}\n"

# 1. Check if mkcert is installed
if ! command -v mkcert &> /dev/null; then
    echo -e "${RED}mkcert is not installed.${NC}"
    echo -e "Please install mkcert before running this script."
    echo -e "Installation instructions:"
    echo -e "  Ubuntu/Debian: sudo apt install libnss3-tools && brew install mkcert (or download binary)"
    echo -e "  Arch Linux:    sudo pacman -S mkcert"
    echo -e "  macOS:         brew install mkcert"
    echo -e "  Windows:       choco install mkcert\n"
    echo -e "Alternatively, download the pre-built binary from: https://github.com/FiloSottile/mkcert/releases"
    exit 1
fi

echo -e "${GREEN}mkcert is installed.${NC}"

# 2. Install the local CA
echo -e "\n${YELLOW}Installing local CA (this may prompt for your password)...${NC}"
mkcert -install

# 3. Detect LAN IP and Hostname
echo -e "\n${YELLOW}Detecting network interfaces...${NC}"
HOSTNAME=$(hostname)

# Try to get LAN IP, fallback to 127.0.0.1 if it fails
if command -v ip &> /dev/null; then
    # Get the default route interface IP
    LAN_IP=$(ip route get 8.8.8.8 | awk -F"src " 'NR==1{split($2,a," ");print a[1]}')
else
    # Fallback for systems without 'ip' command (e.g. some macOS setups)
    LAN_IP=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n 1)
fi

if [ -z "$LAN_IP" ]; then
    LAN_IP="127.0.0.1"
    echo -e "${RED}Could not detect LAN IP, defaulting to 127.0.0.1${NC}"
else
    echo -e "${GREEN}Detected LAN IP: ${LAN_IP}${NC}"
fi
echo -e "${GREEN}Detected Hostname: ${HOSTNAME}${NC}"


# 4. Generate Certificates
mkdir -p certs
echo -e "\n${YELLOW}Generating certificates...${NC}"

# Generate certificate for localhost, 127.0.0.1, ::1, LAN_IP, and Hostname
mkcert \
    -cert-file certs/server.crt \
    -key-file certs/server.key \
    localhost \
    127.0.0.1 \
    ::1 \
    "$LAN_IP" \
    "$HOSTNAME"

echo -e "\n${GREEN}Successfully generated certs/server.crt and certs/server.key${NC}"
echo -e "\nNext Steps:"
echo -e "1. Run 'python app.py'"
echo -e "2. Access the host at https://${LAN_IP}:8000"
echo -e "3. To access from another device on the LAN, you must install the mkcert CA on that device."
echo -e "   Find your root CA by running: mkcert -CAROOT"
echo -e "   Copy rootCA.pem to the other device and install it."
