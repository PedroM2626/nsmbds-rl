#!/usr/bin/env bash
# Build the Rust planner with rustc: cargo.exe is blocked by Smart App Control
# on this machine, and the crate has no dependencies, so rustc is enough.
set -euo pipefail
cd "$(dirname "$0")"
rustc -O --edition 2021 src/main.rs -o nsmb_planner.exe
echo "built $(pwd)/nsmb_planner.exe"
