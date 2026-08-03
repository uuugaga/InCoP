#!/usr/bin/env bash

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/isaacsim/_build/linux-x86_64/release/python.sh}"
TRAJECTORY_ROOT="$PROJECT_ROOT/map/planning/trajectory"
SCENE_ASSET_ROOT="$PROJECT_ROOT/assets/map"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
    echo "Isaac Sim Python was not found or is not executable: $ISAAC_PYTHON" >&2
    echo "Set ISAAC_PYTHON to the Isaac Sim python.sh path." >&2
    exit 1
fi

run_dataset() {
    local scene_dir="$1"
    local condition="$2"
    local usd_file="$3"
    local case_type="${condition#dual_case_}"

    "$ISAAC_PYTHON" "$PROJECT_ROOT/main.py" \
        --condition "$condition" \
        --roadmap "$TRAJECTORY_ROOT/$scene_dir/dual/path_case_$case_type/" \
        --usd_path "$SCENE_ASSET_ROOT/$usd_file"
}

run_dataset "hospital" "dual_case_shadow" "hospital_v1.usd"
run_dataset "hospital" "dual_case_distance" "hospital_v1.usd"
run_dataset "hospital" "dual_case_highoverlap" "hospital_v1.usd"
run_dataset "hospital" "dual_case_random" "hospital_v1.usd"

run_dataset "warehouse" "dual_case_shadow" "full_warehouse_v1.usd"
run_dataset "warehouse" "dual_case_distance" "full_warehouse_v1.usd"
run_dataset "warehouse" "dual_case_highoverlap" "full_warehouse_v1.usd"
run_dataset "warehouse" "dual_case_random" "full_warehouse_v1.usd"

run_dataset "office" "dual_case_shadow" "office_v1.usd"
run_dataset "office" "dual_case_distance" "office_v1.usd"
run_dataset "office" "dual_case_highoverlap" "office_v1.usd"
run_dataset "office" "dual_case_random" "office_v1.usd"
