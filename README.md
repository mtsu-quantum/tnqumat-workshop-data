# TN:QuMat Workshop Data
Data Files for the Hands-On session of TN-QuMat 2025 Workshop

## Getting the data (inside the VM)

```bash
cd
git clone https://github.com/mtsu-quantum/tnqumat-workshop-data.git
cd ~/tnqumat-workshop-data
```

## Running a calculation (inside the VM)
```bash
cd ~/tnqumat-workshop-data/vm_run
mpirun -np 10 main_dca input_sp.json
```

## Analyzing the HPC Data
```bash
cd ~/tnqumat-workshop-data/hpc_data
chmod +x analyze.sh
./analyze.sh
python compute_tc_new.py
```
