#!/bin/bash

echo "CRCNS validation script started."

# Define the parameters
params=(
  "25 1000 25"
  "1025 2000 25"
  "2025 3000 25"
  "3025 4000 25"
  "4025 5000 25"
)

# Loop over the parameters
for param in "${params[@]}"; do
  echo "Starting program with parameters: $param"
  # Start a new instance of the Python program with the current parameters
  python crcns_val_evaluation_all_split.py $param &
  echo "Program with parameters $param finished."
done

# Wait for all background jobs to finish
wait

echo "Script finished."
