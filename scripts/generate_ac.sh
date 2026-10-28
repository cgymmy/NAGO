#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
"$PYTHON_BIN" -m gen_data.ac.generate "$@"
