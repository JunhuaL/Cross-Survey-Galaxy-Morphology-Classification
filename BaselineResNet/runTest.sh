#! /bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/ResnetTest.out

echo $(date)

python baselineResnet.py
