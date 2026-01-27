#!/bin/bash

# Configuration
LOG_DIR="log"
OUTPUT_FILE="parsed_results.txt"

# Clear the output file to start fresh
: > "$OUTPUT_FILE"

# Loop through files in the log directory
for file in "$LOG_DIR"/*; do
    # Skip directories or non-files
    [ -f "$file" ] || continue
    
    filename=$(basename "$file")

    # ---------------------------------------------------------
    # 1. TRAIN FILES: Search for " | Acc: "
    # ---------------------------------------------------------
    if [[ "$filename" == "train_"* ]]; then
        # Find line number of the LAST occurrence of " | Acc: "
        # grep -n adds line numbers; tail -n 1 gets the last one; cut grabs the number
        last_line_num=$(grep -n " | Acc: " "$file" | tail -n 1 | cut -d: -f1)

        if [ -n "$last_line_num" ]; then
            echo "File: $file" | tee -a "$OUTPUT_FILE"
            
            # Print the line found and the one immediately preceding it (context)
            # sed -n 'X,Yp' prints lines from X to Y
            prev_line=$((last_line_num - 1))
            sed -n "${prev_line},${last_line_num}p" "$file" | tee -a "$OUTPUT_FILE"
            
            echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
        fi

    # ---------------------------------------------------------
    # 2. EVAL FILES: Search for starts with "|Tasks|Version|"
    # ---------------------------------------------------------
    elif [[ "$filename" == "eval_"* ]]; then
        # Find line number of the FIRST occurrence of the table header
        target_line_num=$(grep -n "^|Tasks|Version|" "$file" | head -n 1 | cut -d: -f1)

        if [ -n "$target_line_num" ]; then
            echo "File: $file" | tee -a "$OUTPUT_FILE"

            # Calculate range: 
            # 1 line before (config)
            # The target line (header)
            # 2 lines after (separator + values)
            start_line=$((target_line_num - 1))
            end_line=$((target_line_num + 3))

            # Use sed to print exactly that range
            sed -n "${start_line},${end_line}p" "$file" | tee -a "$OUTPUT_FILE"
            
            echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
        fi
    fi
done