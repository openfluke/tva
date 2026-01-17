#!/bin/bash

# Sequence runner for clustering benchmarks (rn1.go, rn2.go, ...)
# Stops if a number in the sequence is missing.

for i in {1..100}
do
    FILE="rn$i.go"
    if [ -f "$FILE" ]; then
        echo "----------------------------------------"
        echo "Running $FILE..."
        echo "----------------------------------------"
        go run "$FILE"
        if [ $? -ne 0 ]; then
            echo "Error: $FILE failed with exit code $?."
            exit 1
        fi
    else
        echo "No $FILE found. Stopping sequence at $i."
        break
    fi
done

echo "Finished sequence."
