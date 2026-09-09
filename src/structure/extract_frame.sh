#!/bin/bash

# Usage: ./extract_and_convert.sh <dump_file> <timestep> <output_data_file>
FILE=$1
TIMESTEP=$2
FINAL_OUT=$3
TEMP_FILE="TEMP_$$.dump" # Using $$ adds the PID to prevent collisions
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dump_file> <timestep> <output_data_file>"
    exit 1
fi

echo "Locating timestep $TIMESTEP..."

# 1. Find the line number of the timestep
MATCH_LINE=$(grep -n -w "^${TIMESTEP}$" "$FILE" | cut -d: -f1)

if [ -z "$MATCH_LINE" ]; then
    echo "Error: Timestep $TIMESTEP not found in $FILE."
    exit 1
fi

START_LINE=$((MATCH_LINE - 1))

# 2. Find the start of the next frame
NEXT_START=$(grep -n "ITEM: TIMESTEP" "$FILE" | awk -v start="$START_LINE" -F: '$1 > start {print $1; exit}')

# 3. Extract to temp file
if [ -z "$NEXT_START" ]; then
    sed -n "${START_LINE},$ p" "$FILE" > "$TEMP_FILE"
else
    END_LINE=$((NEXT_START - 1))
    sed -n "${START_LINE},${END_LINE}p" "$FILE" > "$TEMP_FILE"
fi

# 4. Run the Python conversion
if [ -f "$TEMP_FILE" ]; then
    python3 $SCRIPT_DIR/extract_frame.py "$TEMP_FILE" "$FINAL_OUT"
    
    # 5. Cleanup
    rm "$TEMP_FILE"
    echo "Cleanup complete."
else
    echo "Error: Failed to create temporary dump file."
    exit 1
fi