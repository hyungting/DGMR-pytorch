#!/bin/bash
#SBATCH --job-name="DGMR"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=8
#SBATCH --gres=gpu:2
#SBATCH --time=14-0:0
#SBATCH --chdir=/home/r09521601/dgmr
#SBATCH --output=/home/r09521601/dgmr/log/cout.txt
#SBATCH --error=/home/r09521601/dgmr/log/cerr.txt
echo
echo "============================ Messages from Goddess============================"
echo " * Job starting from: "`date`
echo " * Job ID : "$SLURM_JOBID
echo " * Job name : "$SLURM_JOB_NAME
echo " * Job partition : "$SLURM_JOB_PARTITION
echo " * Nodes : "$SLURM_JOB_NUM_NODES
echo " * Cores : "$SLURM_NTASKS
echo " * Working directory: "${SLURM_SUBMIT_DIR/$HOME/"~"}
echo "==============================================================================="
echo

source ~/mlenv/bin/activate
python3 train.py


echo
echo "============================ Messages from Goddess============================"
echo " * Job ended at : "`date`
echo "==============================================================================="

