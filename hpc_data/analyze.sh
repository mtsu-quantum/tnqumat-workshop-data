#!/bin/bash

# Loop through each dca_tp.hdf5 file in directories starting with T
for file in T*/dca_tp.hdf5; do
    if [[ -f "$file" ]]; then  # Check if the file exists
        echo "Processing $file..."
        python main_analysis.py "$file"  # Run the Python script on each file
    else
        echo "File $file does not exist, skipping."
    fi
done
