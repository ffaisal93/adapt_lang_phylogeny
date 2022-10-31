#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
##SBATCH --output=../tr_output/udp.out
##SBATCH --error=../tr_output/udp.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128; 
#SBATCH --gres=gpu:A100.40gb:3              # up to 8; only request what you need
#SBATCH --mem-per-cpu=35500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=3-01:00:00                   # set to 1hr; please choose carefully

# set echo
# umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

# module load hosts/dgx                        # switch to the modules on the dgx   



base_model=$1
s_lang=$2
task=$3
s_dir=$4
eval=$5
echo ${task}


source ../vnv/vnv-org/bin/activate



if [ "$task" = "udp_mlm" ] && [ "$s_dir" = "uralic" ] && [ "$eval" = "train" ]; then
	# uralic -------------------------------------------------
	output="et_edt"
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test train_mlm \
	--base_model ${base_m} \
	--base_dir ${base_model} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${s_dir}/${s_lang} \
	--lang_name et_1m \
	--task_name et_edt

elif [ "$task" = "pos_mlm" ] && [ "$s_dir" = "uralic" ] && [ "$eval" = "train" ]; then
	# uralic -------------------------------------------------
	output="et_edt"
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_ner.sh \
	--train_test train_mlm \
	--base_model ${base_m} \
	--base_dir ${base_model} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${s_dir}/${s_lang} \
	--lang_name et_1m \
	--task_name et_edt


elif [ "$task" = "udp_mlm" ] && [ "$eval" = "udp_eval" ]; then
	# uralic -------------------------------------------------
	output="et_edt"
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test udp_eval \
	--base_model ${base_m} \
	--base_dir ${base_model} \
	--data_dir ../data/uralic_demo \
	--out_dir ${base_model} \
	--task_name mlm \
	--lang_config ../meta_files/mlm_info.json \
	--family_name ${s_dir}


elif [ "$task" = "udp_mlm" ] && [ "$s_dir" = "germanic" ]; then
	# uralic -------------------------------------------------
	output="et_edt"
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test train_mlm \
	--base_model ${base_m} \
	--base_dir ${base_model} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${s_dir}/${s_lang} \
	--lang_name et_1m \
	--task_name en_ewt

elif [ "$task" = "udp_mlm" ] && [ "$s_dir" = "tupian" ]; then
	# uralic -------------------------------------------------
	output="et_edt"
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test train_mlm \
	--base_model ${base_m} \
	--base_dir ${base_model} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${s_dir}/${s_lang} \
	--lang_name et_1m \
	--task_name en_ewt
	# base_m='mbert'
	# base_model=
	# ./run_udp.sh \
	# --train_test train_mlm \
	# --base_model ${base_m} \
	# --base_dir ../adapters/${base_m}/germanic-mlm/en_1m \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${s_lang} \
	# --lang_name en_1m \
	# --task_name en_ewt

	# output="en_ewt"
	# base_m='mbert'
	# base_mh='bert-base-multilingual-cased'
	# ./run_udp.sh \
	# --train_test train_mlm \
	# --base_model ${base_m} \
	# --base_dir ../adapters/${base_m}/germanic-mlm/en_1m \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --lang_name en_1m \
	# --task_name en_ewt

fi



