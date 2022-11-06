#!/bin/bash
#SBATCH -J ASR
#SBATCH -w gpuc1
#SBATCH -o ./run_out
#SBATCH -e ./run_err
#SBATCH -p gpu
#SBATCH --gres=gpu:1


##python /home/yupei/workspaces/ASR/src/filepath_timit.py

##python /home/yupei/workspaces/ASR/src/feature_timit.py

##python /home/yupei/workspaces/ASR/src/target_timit.py

python /home/yupei/workspaces/ASR/src/main_MND_gru_ctc.py

##python /home/yupei/workspaces/ASR/src/main_MND_gru.py


