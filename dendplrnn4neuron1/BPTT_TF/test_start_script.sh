#!/bin/bash

echo "CRCNS validation --TEST-- script started."

# Define the parameters
params=(
  "25 501 25"
  "525 1001 25"
  "1025 1501 25"
  "1525 2001 25"
  "2025 2501 25"
  "2525 3001 25"
  "3025 3501 25"
  "3525 4001 25"
  "4025 4501 25"
  "4525 5001 25"
)

# Loop over the parameters
for param in "${params[@]}"; do
  echo "Starting program with parameters: $param"
  # Start a new instance of the Python program with the current parameters
  python crcns_test_evaluation_all_split.py $param &
  echo "Program with parameters $param finished."
done

# Wait for all background jobs to finish
wait

echo "Script finished."
