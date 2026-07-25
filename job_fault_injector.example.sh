#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # <- one task: my_model.py is serial and not MPI-aware
#SBATCH --cpus-per-task=16   # <- CPU cores for the job; also sets OMP_NUM_THREADS below
#SBATCH --partition=gpuA40x4      # <- GPU partition; lightning.gpu needs compute capability >= 7.0
#SBATCH --account=<YOUR_ACCOUNT>    # <- GPU allocation; run `accounts` to list yours (e.g. xyz123-delta-gpu)
#SBATCH --job-name=fault_injector
#SBATCH --time=01:00:00      # hh:mm:ss for the job; raise this if your run is longer
#SBATCH --constraint="scratch&work"
#SBATCH -e slurm-fault_injector-%j.err
#SBATCH -o slurm-fault_injector-%j.out
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load miniforge3-python  # ... or any appropriate modules
module list  # job documentation and metadata
source <PATH_TO_YOUR_VENV>/bin/activate
which python

# my_model.py imports qml_fault_injector, which is NOT pip-installed — it
# resolves off the script's own directory, so run from here.
cd <PATH_TO_REPO>/fault_injector

srun python <MODEL FILE NAME>.py
