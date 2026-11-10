#!/bin/sh
# GENESIS one-command installer (Linux, macOS, WSL):
#
#   curl -fsSL https://github.com/ivan-14-dev/genesis/releases/latest/download/install.sh | sh
#
# It installs the latest release into its own virtual environment (~/.genesis/venv), puts a `genesis` command in
# ~/.local/bin, runs `genesis init` (admin key, memory and web credentials generated on this machine, printed once
# and saved in ~/.genesis/credentials.txt) and opens the web interface. Options (after `sh -s --` when piped):
#
#   --version X.Y.Z   install that release instead of the latest      (env: GENESIS_VERSION)
#   --home DIR        GENESIS folder, default ~/.genesis              (env: GENESIS_HOME)
#   --port N          web interface port, default 8766                (env: GENESIS_PORT)
#   --bin-dir DIR     where to link the `genesis` command, default ~/.local/bin
#   --source SRC      install from a wheel, a source folder or a URL instead of the GitHub release
#   --no-start        do not start the web interface at the end
#
# Nothing is installed outside the GENESIS folder and the bin folder; GENESIS has no third-party dependency.
set -eu

say() { printf '%s\n' "$*"; }
die() { printf 'genesis install: %s\n' "$*" >&2; exit 1; }

main() {
  REPO="ivan-14-dev/genesis"
  VERSION="${GENESIS_VERSION:-latest}"
  HOME_DIR="${GENESIS_HOME:-$HOME/.genesis}"
  PORT="${GENESIS_PORT:-8766}"
  BIN_DIR="$HOME/.local/bin"
  SOURCE=""
  START=1

  while [ $# -gt 0 ]; do
    case "$1" in
      --version) [ $# -ge 2 ] || die "--version needs a value"; VERSION="$2"; shift 2 ;;
      --home) [ $# -ge 2 ] || die "--home needs a value"; HOME_DIR="$2"; shift 2 ;;
      --port) [ $# -ge 2 ] || die "--port needs a value"; PORT="$2"; shift 2 ;;
      --bin-dir) [ $# -ge 2 ] || die "--bin-dir needs a value"; BIN_DIR="$2"; shift 2 ;;
      --source) [ $# -ge 2 ] || die "--source needs a value"; SOURCE="$2"; shift 2 ;;
      --no-start) START=0; shift ;;
      -h|--help) sed -n '2,17p' "$0" 2>/dev/null || true; exit 0 ;;
      *) die "unknown option: $1" ;;
    esac
  done
  case "$PORT" in ''|*[!0-9]*) die "--port must be a number" ;; esac

  PYTHON=""
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
        >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
  [ -n "$PYTHON" ] || die "Python 3.10 or newer is required (https://www.python.org/downloads/)"
  "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1 \
    || die "the Python 'venv' module is missing (Debian/Ubuntu: sudo apt install python3-venv)"

  if [ -z "$SOURCE" ]; then
    say "Looking up GENESIS release: $VERSION"
    SOURCE=$("$PYTHON" - "$REPO" "$VERSION" <<'PY'
import json, sys, urllib.request
repo, version = sys.argv[1], sys.argv[2]
tag = version if version in ("latest",) or version.startswith("v") else "v" + version
api = f"https://api.github.com/repos/{repo}/releases/" + ("latest" if tag == "latest" else f"tags/{tag}")
req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json", "User-Agent": "genesis-install"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        release = json.load(r)
except Exception as exc:
    sys.exit(f"cannot read {api}: {exc}")
wheels = [a["browser_download_url"] for a in release.get("assets", []) if a["name"].endswith(".whl")]
if not wheels:
    sys.exit(f"release {release.get('tag_name')} has no wheel asset")
print(wheels[0])
PY
  ) || die "could not find a GENESIS release to install"
  fi

  say "Installing GENESIS into $HOME_DIR"
  mkdir -p "$HOME_DIR"
  chmod 700 "$HOME_DIR"
  VENV="$HOME_DIR/venv"
  [ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV" || die "could not create the virtual environment $VENV"
  PIP_DISABLE_PIP_VERSION_CHECK=1 "$VENV/bin/python" -m pip install --quiet --upgrade --force-reinstall "$SOURCE" \
    || die "pip could not install $SOURCE"

  mkdir -p "$BIN_DIR"
  ln -sf "$VENV/bin/genesis" "$BIN_DIR/genesis"
  "$VENV/bin/genesis" --version

  say ""
  "$VENV/bin/genesis" init --home "$HOME_DIR" --port "$PORT"
  say ""
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) say "Add $BIN_DIR to your PATH to use the 'genesis' command, e.g.:"
       say "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.profile" ;;
  esac
  if [ "$START" -eq 1 ]; then
    say "Starting the web interface (Ctrl+C to stop; later: genesis start --open)"
    exec "$VENV/bin/genesis" start --home "$HOME_DIR" --port "$PORT" --open
  fi
  say "Done. Start it with: genesis start --open"
}

main "$@"
