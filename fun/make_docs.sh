#!/bin/bash

# Output file
OUTPUT_FILE="FULL_REPORT.md"

# Clear/Create output file
echo "" > "$OUTPUT_FILE"

# 1. Add README.md
if [ -f "README.md" ]; then
    echo "Processing README.md..."
    cat README.md >> "$OUTPUT_FILE"
    echo -e "\n\n---\n" >> "$OUTPUT_FILE"
else
    echo "Warning: README.md not found."
fi

# 2. Add all .go files recursively
echo "Processing .go files..."

# Find all .go files, sort them to be deterministic
find . -type f -name "*.go" | sort | while read -r filepath; do
    # Skip the output file itself if it happened to be .go (unlikely) and skip temp files
    
    echo "Adding $filepath..."
    
    # Add Header
    echo -e "\n# File: $filepath" >> "$OUTPUT_FILE"
    echo '```go' >> "$OUTPUT_FILE"
    
    # Add Content
    cat "$filepath" >> "$OUTPUT_FILE"
    
    # Close Block
    echo -e "\n\`\`\`" >> "$OUTPUT_FILE"
    echo -e "\n---\n" >> "$OUTPUT_FILE"
done

echo "Done! Documentation generated at $OUTPUT_FILE"
