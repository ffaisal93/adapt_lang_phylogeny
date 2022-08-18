#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
#SBATCH --output=../tr_output/ie_mlm.out
#SBATCH --error=../tr_output/ie_mlm.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:4              # up to 8; only request what you need
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

cd ../
source vnv/ie_train/bin/activate

python new_scripts/run_mlm_ie.py \
        --model_name_or_path  /scratch/ffaisal/base_models/mbert \
        --train_files data/ie \
        --lang_config meta_files/lang_meta_ie.json \
        --lang_family ie \
        --do_train \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --output_dir adapters/mbert/ie\
        --train_adapter \
        --cache_dir /scratch/ffaisal/hug_cache \
        --adapter_config "pfeiffer+inv" \
        --max_seq_length 512 \
        --save_steps 30000 \
        --per_device_train_batch_size 10 \
        --adapter_reduction_factor 16 \
        --overwrite_output_dir
