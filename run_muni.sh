#!/bin/bash
# Remove previous results
rm -f muni_results.txt

# Run main test with arguments
echo "Running muniversal_testing.go with args: $@"
go run muniversal_testing.go "$@" >> muni_results.txt 2>&1

# Display results
cat muni_results.txt
