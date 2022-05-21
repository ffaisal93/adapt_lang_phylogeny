#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
##SBATCH --output=../tr_output/h_temp_tup.out
##SBATCH --error=../tr_output/h_temp_tup.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:4              # up to 8; only request what you need
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully

# set echo
# umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

# module load hosts/dgx                        # switch to the modules on the dgx   






# file='../mono-data/af1.txt'
# name='dummy'

# cd ../GENRE
# python genre_pr.py $file $name

# source ~/.bashrc
# conda activate transfer

# module load cuda/10.2

# module load singularity
# SINGULARITY_BASE=/containers/dgx/Containers
# singularity run --nv -B ${PWD}:/scratch/ffaisal/run_mlm/slurm_scripts --pwd /scratch/ffaisal/run_mlm/slurm_scripts ${SINGULARITY_BASE}/pytorch/pytorch_21.02-py3.sif /scratch/ffaisal/run_mlm/slurm_scripts/temp.sh

# export HF_DATASETS_CACHE='/scratch/ffaisal/hug_cache/datasets'

source /projects/antonis/fahim/venvs/hopper/adapt/bin/activate

file=$1
name=$2
outname=$3

echo $1
echo $2
echo $3

python ../adapter-transformers/examples/language-modeling/run_mlm.py \
    --model_name_or_path  /scratch/ffaisal/base_models/mbert \
    --train_file ${file} \
    --do_train \
    --learning_rate 1e-4 \
    --num_train_epochs 40 \
    --output_dir ../adapters/mbert/${outname}_l/${name} \
    --train_adapter \
    --cache_dir /scratch/ffaisal/hug_cache/datasets \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --save_steps 5000 \
    --per_device_train_batch_size 12 \
    --adapter_config "pfeiffer+inv"




######xlmr
# python ../adapter-transformers/examples/language-modeling/run_mlm.py \
#     --model_name_or_path  /scratch/ffaisal/base_models/xlmr \
#     --train_file ${file} \
#     --do_train \
#     --learning_rate 2e-5 \
#     --num_train_epochs 3 \
#     --output_dir ../adapter/${outname}/${name} \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache/datasets \
#     --overwrite_output_dir \
#     --max_seq_length 256 \
#     --save_steps 5000 \
#     --per_device_train_batch_size 8 \
#     --adapter_config "pfeiffer+inv"

#---------------mlm training---------------------------------
# file=$1
# name=$2

# echo $1
# echo $2

# python ../adapter-transformers/examples/language-modeling/run_mlm.py \
#     --model_name_or_path  /scratch/ffaisal/base_models/xlmr \
#     --train_file ../americas/family.txt \
#     --do_train \
#     --learning_rate 2e-5 \
#     --num_train_epochs 40 \
#     --output_dir ../adapter/americas_nli_mlm \
#     --cache_dir /scratch/ffaisal/hug_cache/datasets \
#     --overwrite_output_dir \
#     --max_seq_length 256 \
#     --save_steps 5000 \
#     --per_device_train_batch_size 8 


