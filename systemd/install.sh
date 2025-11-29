#!/bin/bash
# Install music-models systemd user units

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_UNITS="$HOME/.config/systemd/user"

echo "Installing music-models systemd units..."

mkdir -p "$USER_UNITS"

for service in "$SCRIPT_DIR"/*.service; do
    name=$(basename "$service")
    echo "  Installing $name"
    cp "$service" "$USER_UNITS/"
done

systemctl --user daemon-reload

echo ""
echo "✓ Installed $(ls "$SCRIPT_DIR"/*.service | wc -l) service units"
echo ""
echo "Usage:"
echo "  systemctl --user start clap       # Start a service"
echo "  systemctl --user enable clap      # Enable on boot"
echo "  systemctl --user status clap      # Check status"
echo "  journalctl --user -u clap -f      # View logs"
