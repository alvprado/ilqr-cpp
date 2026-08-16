#!/usr/bin/env bash
set -euo pipefail

EXAMPLES_DIR=/workspace/build/examples

list_examples() {
    for example_dir in "$EXAMPLES_DIR"/*/; do
        example_name="$(basename "$example_dir")"
        [ -x "$example_dir$example_name" ] && echo "  - $example_name"
    done
}

usage() {
    echo "Usage: docker run --rm -it <image> <example>"
    echo "Available examples:"
    list_examples
}

if [ "$#" -eq 0 ]; then
    exec /bin/bash
fi

example_name="$1"
example_binary="$EXAMPLES_DIR/$example_name/$example_name"
if [ ! -x "$example_binary" ]; then
    echo "Unknown example: $example_name" >&2
    usage >&2
    exit 1
fi

shift
if [ -t 0 ]; then
    # Interactive run 
    "$example_binary" "$@" || true
    exec /bin/bash
else
    # Non-interactive run
    exec "$example_binary" "$@"
fi
