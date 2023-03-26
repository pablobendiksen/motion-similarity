#!/bin/bash
#SBATCH --job-name=array_parallelized_deepsight
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=p.bendiksen001@umb.edu # Where to send mail
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12, chimera13
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu:1
#SBATCH -export=NONE # Export no environment variables
#SBATCH --mem=8gb
#SBATCH -t 14-6:30:00
#SBATCH --output=/home/p.bendiksen001/array/%A-%a.out
#SBATCH --error=/home/p.bendiksen001/array/%A-%a.err
#SBATCH --array=1-20

. /etc/profile

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate motion-similarity

python main.py $SLURM_ARRAY_TASK_ID

echo "Finish Run"