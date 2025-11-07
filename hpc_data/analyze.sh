#!/bin/bash
source ~/alps-venv/bin/activate
pip install h5py pandas plotnine

# Loop through each dca_tp.hdf5 file in directories starting with T
for file in T*/dca_tp.hdf5; do
    if [[ -f "$file" ]]; then  # Check if the file exists
        echo "Processing $file..."
        python analyze_main.py "$file"  # Run the Python script on each file
    else
        echo "File $file does not exist, skipping."
    fi
done
