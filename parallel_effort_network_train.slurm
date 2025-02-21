#!/bin/bash
#SBATCH --job-name=array_parallelized_motion
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=b.bendiksen001@umb.edu # Where to send mail
##SBATCH -p Intel6240,Intel6248,DGXA100
#SBATCH -p DGXA100
#SBATCH -A pi_funda.durupinarbabur
#SBATCH --qos=scavenger
#SBATCH -w chimera12
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu:1
#SBATCH -export=NONE # Export no environment variables
#SBATCH --mem=20G
#SBATCH -t 3-23:59:59
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion_sim_arrays/%A-%a.out
#SBATCH --error=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion_sim_arrays/%A-%a.err
#SBATCH --array=0-4

. /etc/profile

# check cpu number per task, should be equal to -n
export = TF_USE_LEGACY_KERAS=1
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate gpu_env

python main.py $SLURM_ARRAY_TASK_ID
echo "Finish Run"

