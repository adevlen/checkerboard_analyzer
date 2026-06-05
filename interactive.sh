#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building checkerboard_container image..."
cd "$SCRIPT_DIR/.devcontainer"
docker build -t checkerboard_container .

echo ""
echo "Starting interactive shell (project mounted at /work)..."
echo "  python demo.py --input ZR751_test.xlsx"
echo "  pytest -v tests/tests.py"
echo "  exit   # leave the container"
echo ""

docker run -it --rm \
    -v "$SCRIPT_DIR:/work" \
    -w /work \
    checkerboard_container \
    bash
