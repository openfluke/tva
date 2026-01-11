#!/bin/bash

# Change directory to where the script is located
cd "$(dirname "$0")"

# Find all .go files in the current directory (tva/fun) and subdirectories
find . -type f -name "*.go" | sort | while read -r file; do
    echo "==============================================================================="
    echo "RUNNING: $file"
    echo "==============================================================================="
    
    # Run the go file
    go run "$file"
    
    echo ""
    echo "Finished $file"
    echo ""
    sleep 1
done

echo "All tests completed."
