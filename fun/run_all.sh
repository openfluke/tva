#!/bin/bash

# Change directory to where the script is located
cd "$(dirname "$0")"

OUTPUT_FILE="results.txt"

# Clear previous results
> "$OUTPUT_FILE"

echo "Starting all tests... Output will be saved to $OUTPUT_FILE"
echo "Starting all tests at $(date)" >> "$OUTPUT_FILE"

# Find all .go files in the current directory (tva/fun) and subdirectories
# Sort to ensure consistent order (ftest1, ftest2, ...)
find . -type f -name "*.go" | sort | while read -r file; do
    echo "===============================================================================" | tee -a "$OUTPUT_FILE"
    echo "RUNNING: $file" | tee -a "$OUTPUT_FILE"
    echo "===============================================================================" | tee -a "$OUTPUT_FILE"
    
    # Run the go file, append output to OUTPUT_FILE and show on screen
    # 2>&1 redirects stderr to stdout so we capture errors too
    go run "$file" 2>&1 | tee -a "$OUTPUT_FILE"
    
    echo "" | tee -a "$OUTPUT_FILE"
    echo "Finished $file" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    sleep 2
done

echo "All tests completed." | tee -a "$OUTPUT_FILE"
