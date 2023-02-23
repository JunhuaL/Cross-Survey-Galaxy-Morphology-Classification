#! /bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/EncoderOutput.out

echo $(date)

python autoEncoder.py
