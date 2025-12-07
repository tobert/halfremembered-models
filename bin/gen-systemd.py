#!/usr/bin/env python3
"""Generate systemd user units for halfremembered music model services.

Usage:
    ./bin/gen-systemd.py clap              # Print unit to stdout
    ./bin/gen-systemd.py --all             # Print all units
    ./bin/gen-systemd.py --all -o systemd/ # Write all to directory
    ./bin/gen-systemd.py --list            # List available services
    ./bin/gen-systemd.py --verify          # Verify all units with systemd-analyze
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

# Service definitions: name -> (port, description)
SERVICES: dict[str, tuple[int, str]] = {
    "orpheus-base": (2000, "Orpheus Base MIDI Generation"),
    "orpheus-classifier": (2001, "Orpheus AI vs Human Classifier"),
    "orpheus-bridge": (2002, "Orpheus Musical Bridge Generation"),
    "orpheus-loops": (2003, "Orpheus Loop Generation"),
    "orpheus-children": (2004, "Orpheus Children's Music Generation"),
    "orpheus-mono": (2005, "Orpheus Monophonic Melody Generation"),
    "musicgen": (2006, "Meta MusicGen Text-to-Music"),
    "clap": (2007, "CLAP Audio Analysis and Embeddings"),
    "yue": (2008, "YuE Text-to-Song Generation"),
    "stable-audio": (2009, "Stability AI Audio Generation"),
    "audioldm2": (2010, "AudioLDM2 Audio Generation"),
    "anticipatory": (2011, "Anticipatory Music Generation"),
    "beat-this": (2012, "Beat This! Beat and Downbeat Tracking"),
    "llmchat": (2020, "OpenAI-compatible LLM with Tool Calling"),
    "observer": (2099, "ROCm GPU Observability Agent"),
}

# Extra environment variables per service (for service-specific config)
SERVICE_ENV: dict[str, list[str]] = {
    "llmchat": ["LLMCHAT_MODEL=qwen3-vl-4b"],
}

UNIT_TEMPLATE = """\
[Unit]
Description={description}
After=network.target

[Service]
Type=exec
WorkingDirectory={repo_path}/services/{service}
ExecStart={uv_path} run python server.py
Restart=on-failure
RestartSec=10

Environment=PYTHONUNBUFFERED=1
Environment=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
Environment=OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
{extra_env}
StandardOutput=journal
StandardError=journal
SyslogIdentifier={service}

NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
"""


def find_repo_root() -> Path:
    """Find the repository root by looking for CLAUDE.md."""
    path = Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "CLAUDE.md").exists():
            return path
        path = path.parent
    raise RuntimeError("Could not find repository root (no CLAUDE.md found)")


def find_uv() -> Path:
    """Find the uv binary path."""
    # Check common locations
    candidates = [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
        Path("/usr/bin/uv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fall back to just "uv" and let systemd find it
    return Path("uv")


def generate_unit(service: str, repo_path: Path, uv_path: Path) -> str:
    """Generate a systemd unit file for a service."""
    if service not in SERVICES:
        raise ValueError(f"Unknown service: {service}")

    port, description = SERVICES[service]

    # Build extra environment lines
    extra_env_lines = SERVICE_ENV.get(service, [])
    extra_env = "\n".join(f"Environment={env}" for env in extra_env_lines)

    return UNIT_TEMPLATE.format(
        service=service,
        description=description,
        repo_path=repo_path,
        uv_path=uv_path,
        port=port,
        extra_env=extra_env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate systemd user units for music model services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
              %(prog)s clap                  # Print clap unit to stdout
              %(prog)s --all                 # Print all units to stdout
              %(prog)s --all -o systemd/     # Write all units to systemd/
              %(prog)s --list                # List available services
        """),
    )
    parser.add_argument(
        "service",
        nargs="?",
        help="Service name to generate unit for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate units for all services",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available services and exit",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Generate units to temp dir and verify with systemd-analyze",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Write units to directory instead of stdout",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available services:")
        for name, (port, desc) in sorted(SERVICES.items(), key=lambda x: x[1][0]):
            print(f"  {name:20} port {port:5}  {desc}")
        return 0

    # Verify mode - uses tempfile.TemporaryDirectory for safe cleanup
    if args.verify:
        repo_path = find_repo_root()
        uv_path = find_uv()
        failed = False

        with tempfile.TemporaryDirectory(prefix="systemd-verify-") as tmpdir:
            tmppath = Path(tmpdir)
            print("Verifying systemd units...")

            for service in SERVICES:
                unit_content = generate_unit(service, repo_path, uv_path)
                unit_path = tmppath / f"{service}.service"
                unit_path.write_text(unit_content)

                result = subprocess.run(
                    ["systemd-analyze", "verify", str(unit_path)],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0 or result.stderr.strip():
                    print(f"  ✗ {service}")
                    if result.stderr:
                        print(f"    {result.stderr.strip()}")
                    failed = True
                else:
                    print(f"  ✓ {service}")

        return 1 if failed else 0

    # Validate args
    if not args.all and not args.service:
        parser.error("Either specify a service name or use --all")

    if args.service and args.service not in SERVICES:
        print(f"Error: Unknown service '{args.service}'", file=sys.stderr)
        print(f"Use --list to see available services", file=sys.stderr)
        return 1

    # Find paths
    repo_path = find_repo_root()
    uv_path = find_uv()

    # Determine which services to generate
    services = list(SERVICES.keys()) if args.all else [args.service]

    # Generate units
    for service in services:
        unit_content = generate_unit(service, repo_path, uv_path)

        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            unit_path = args.output_dir / f"{service}.service"
            unit_path.write_text(unit_content)
            print(f"  {unit_path}")
        else:
            if args.all:
                print(f"# === {service}.service ===")
            print(unit_content)

    return 0


if __name__ == "__main__":
    sys.exit(main())
