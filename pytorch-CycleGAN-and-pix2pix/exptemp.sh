#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --partition=long
#SBATCH --error=/home/mila/a/alexia.reynaud/hackathon/cyclegan_error.txt
#SBATCH --output=/home/mila/a/alexia.reynaud/omnigan/job_output.txt
#SBATCH -o /home/mila/a/alexia.reynaud/omnigan/slurm-%j.out

cd /home/mila/a/alexia.reynaud/hackathon/weather_events_poc/pytorch-CycleGAN-and-pix2pix/

# 1. Load your environment
echo "Starting job"
module load anaconda/3 >/dev/null 2>&1
. "$CONDA_ACTIVATE"
conda activate python_env

# 2. Launch your job
python3 test.py --dataroot ./datasets/houses_fire --name houses_fire_cyclegan --model cycle_gan
echo 'done'
