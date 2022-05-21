#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
#SBATCH --output=../tr_output/xlmr_ger.out
#SBATCH --error=../tr_output/xlmr_ger.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:3              # up to 8; only request what you need
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

# module load hosts/dgx                        # switch to the modules on the dgx   





source ../../vnv/bin/activate



# ./run_mlm_all.sh \
# 	--train_test train_joint \
# 	--base_model xlmr \
# 	--base_dir /scratch/ffaisal/base_models/xlmr \
# 	--data_dir ../data/germanic_up \
# 	--out_dir ../adapters/xlmr/germanic_up \
# 	--lang_family germanic 

./run_mlm_all.sh \
	--train_test train_joint \
	--base_model mbert \
	--base_dir /scratch/ffaisal/base_models/mbert \
	--data_dir ../data/uto_aztecan \
	--out_dir ../adapters/mbert/uto_aztecan_fgl-j \
	--lang_family uto_aztecan 

# ./run_mlm_all.sh \
# 	--train_test train_joint \
# 	--base_model xlmr \
# 	--base_dir /scratch/ffaisal/base_models/xlmr \
# 	--data_dir ../data/uto_aztecan \
# 	--out_dir ../adapters/xlmr/uto_aztecan \
# 	--lang_family uto_aztecan 
