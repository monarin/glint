#!/bin/bash
#SBATCH --job-name=glint-ab
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/scratch/users/m/monarin/gpu-validation/glint-ab/job-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/lscratch/monarin/glint-ab-${SLURM_JOB_ID}
export CUPY_CACHE_DIR=/sdf/scratch/users/m/monarin/gpu-validation/glint-ab/cache-${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$CUPY_CACHE_DIR"
cd "$STAGE_AB_SOURCE"
/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/ps_20241122/bin/python experiments/xtc_bridge/benchmark_psana_ab.py --config "$STAGE_AB_CONFIG" --output /sdf/scratch/users/m/monarin/gpu-validation/glint-ab/run-${SLURM_JOB_ID} --revision "$STAGE_AB_REVISION" --pairs 3 --timeout 2100
