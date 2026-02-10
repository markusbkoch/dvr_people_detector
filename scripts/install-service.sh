#!/bin/bash
# Install the DVR People Detector as a systemd service
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATE_FILE="$PROJECT_DIR/dvr-detector.service.template"
SERVICE_FILE="/etc/systemd/system/dvr-detector.service"

if [[ $EUID -ne 0 ]]; then
   echo "Run with sudo: sudo $0" 
   exit 1
fi

if [[ ! -f "$TEMPLATE_FILE" ]]; then
   echo "Template not found: $TEMPLATE_FILE"
   exit 1
fi

# Get the user who owns the project directory
PROJECT_USER=$(stat -c '%U' "$PROJECT_DIR")

echo "Installing dvr-detector.service..."
echo "  Project dir: $PROJECT_DIR"
echo "  User: $PROJECT_USER"

# Generate service file from template
sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__USER__|$PROJECT_USER|g" \
    "$TEMPLATE_FILE" > "$SERVICE_FILE"

systemctl daemon-reload
systemctl enable dvr-detector

echo ""
echo "Service installed. Commands:"
echo "  sudo systemctl start dvr-detector   # Start"
echo "  sudo systemctl stop dvr-detector    # Stop"
echo "  sudo systemctl status dvr-detector  # Status"
echo "  journalctl -u dvr-detector -f       # Follow logs"
