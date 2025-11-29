# Systemd Units

This directory is for generated systemd unit files. Units are **not** committed
to git - they're generated dynamically based on the current repo location.

## Usage

```bash
# Generate units to this directory (for inspection)
just systemd-gen-all

# Generate and install directly to ~/.config/systemd/user/
just systemd-install

# Generate a single unit to stdout
just systemd-gen clap
```

## Why Generated?

The unit files contain absolute paths to the repository. Generating them
dynamically means:
- No stale paths after moving the repo
- No hardcoded usernames or home directories
- Single source of truth for service definitions (bin/gen-systemd.py)
