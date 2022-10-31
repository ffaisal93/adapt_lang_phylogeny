#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
#SBATCH --output=../tr_output/udp.out
#SBATCH --error=../tr_output/udp.err
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





# source ../vnv/vnv-org/bin/activate
source ../vnv/vnv-adp-l/bin/activate
# pip install -r ../adapter-transformers-l/examples/pytorch/dependency-parsing/requirements.txt

export TASK_NAME="en_ewt"

python ../adapter-transformers-l/examples/pytorch/dependency-parsing/run_udp.py \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_predict \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 12 \
    --learning_rate 5e-4 \
    --num_train_epochs 1 \
    --max_seq_length 256 \
    --output_dir ../experiments/$TASK_NAME \
    --overwrite_output_dir \
    --store_best_model \
    --evaluation_strategy epoch \
    --metric_score uas 



# ./run_udp.sh \
# 	--train_test train_mlm \
# 	--base_model mbert \
# 	--base_dir ../adapters/mbert/germanic-mlm/en_1m \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/mbert/udp_mlm/germanic/en_1m \
# 	--lang_name et_1m \
# 	--task_name en_ewt

# ./run_udp.sh \
# --train_test udp_eval \
# --base_model mbert \
# --base_dir ../experiments/mbert/udp_mlm/uralic/et_1m \
# --data_dir ../data/uralic_demo \
# --out_dir ../experiments/mbert/udp_mlm/uralic/et_1m \
# --task_name mlm \
# --lang_config ../meta_files/mlm_info.json \
# --family_name uralic_up

#/scratch/ffaisal/base_models/bert-base-multilingual-cased
#../adapters/mbert/germanic-mlm/en_1m \
#../experiments/mbert/udp_mlm/germanic/en_1m \