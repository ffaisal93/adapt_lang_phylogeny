#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
#SBATCH --output=../tr_output/udp.out
#SBATCH --error=../tr_output/udp.err
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



task=$1
echo ${task}


source ../vnv/vnv-org/bin/activate



if [ "$task" = "udp_mlm" ]; then
	## uralic -------------------------------------------------
	# output="et_edt"
	# base_m='mbert'
	# base_mh='bert-base-multilingual-cased'
	# ./run_udp.sh \
	# --train_test train_mlm \
	# --base_model ${base_m} \
	# --base_dir ../adapters/${base_m}/uralic-mlm/et_1m \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --lang_name et_1m \
	# --task_name et_edt

	output="en_ewt"
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test train_mlm \
	--base_model ${base_m} \
	--base_dir ../adapters/${base_m}/germanic-mlm/en_1m \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_name en_1m \
	--task_name en_ewt

	output="en_ewt"
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test train_mlm \
	--base_model ${base_m} \
	--base_dir ../adapters/${base_m}/germanic-mlm/en_1m \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_name en_1m \
	--task_name en_ewt

fi

if [ "$task" = "udp_lll" ]; then
	## uralic -------------------------------------------------
	output="et_edt_lll-j"
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_mh} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/uralic_st/et_1m/mlm1 \
	--group_path ../adapters/${base_m}/uralic_st/et_1m/mlm2 \
	--lang_path ../adapters/${base_m}/uralic_st/et_1m/mlm3 \
	--lang_name et_1m \
	--task_name et_edt

fi

if [ "$task" = "udp" ]; then
	## uralic -------------------------------------------------
	output="et_edt_fgl-j"
	base_m='xlmr'
	./run_udp.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/uralic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/uralic_up_fgl-j/finnic \
	--lang_path ../adapters/${base_m}/uralic_up_fgl-j/et_1m \
	--lang_name et_1m \
	--task_name et_edt

	base_m='mbert'
	./run_udp.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/uralic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/uralic_up_fgl-j/finnic \
	--lang_path ../adapters/${base_m}/uralic_up_fgl-j/et_1m \
	--lang_name et_1m \
	--task_name et_edt


	##train-lt
	output="et_edt_l"
	base_m='xlmr'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/uralic_up_l/et_1m/mlm \
	--lang_name et_1m \
	--task_name et_edt

	base_m='mbert'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/uralic_up_l/et_1m/mlm \
	--lang_name et_1m \
	--task_name et_edt




	## germanic ------------------------------------------------
	output="en_ewt_fgl-j"
	base_m='xlmr'
	./run_udp.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
	--lang_name en_1m \
	--task_name en_ewt

	output="en_ewt_fgl-j"
	base_m='mbert'
	./run_udp.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
	--lang_name en_1m \
	--task_name en_ewt

	##train-lt
	output="en_ewt_l"
	base_m='xlmr'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
	--lang_name en_1m \
	--task_name en_ewt

	output="en_ewt_l"
	base_m='mbert'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
	--lang_name en_1m \
	--task_name en_ewt

	# rm -rf ../experiments/xlmr/${task}/en_ewt/best_model
	# rm -rf ../experiments/xlmr/${task}/en_ewt/checkpoint*

elif [ "$task" = "pos" ]; then
	## uralic -------------------------------------------------
	##train_joint----xlmr
	output="et_edt_fgl-j"
	base_m='xlmr'
	./run_ner.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/uralic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/uralic_up_fgl-j/finnic \
	--lang_path ../adapters/${base_m}/uralic_up_fgl-j/et_1m \
	--lang_name et_1m \
	--task_name et_edt

	##train_joint----mbert
	base_m='mbert'
	./run_ner.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/uralic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/uralic_up_fgl-j/finnic \
	--lang_path ../adapters/${base_m}/uralic_up_fgl-j/et_1m \
	--lang_name et_1m \
	--task_name et_edt

	output="et_edt_l"
	##train_lt----xlmr
	base_m='xlmr'
	./run_ner.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/uralic_up_l/et_1m/mlm \
	--lang_name et_1m \
	--task_name et_edt

	##train_lt----mbert
	base_m='mbert'
	./run_ner.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/uralic_up_l/et_1m/mlm \
	--lang_name et_1m \
	--task_name et_edt



	## germanic ------------------------------------------------
	output="en_ewt_fgl-j"
	#joint
	base_m='xlmr'
	./run_ner.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
	--lang_name en_1m \
	--task_name en_ewt

	base_m='mbert'
	./run_ner.sh \
	--train_test train_joint \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
	--lang_name en_1m \
	--task_name en_ewt

	#lt
	output="en_ewt_l"
	base_m='xlmr'
	./run_ner.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
	--lang_name en_1m \
	--task_name en_ewt

	base_m='mbert'
	./run_ner.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
	--lang_name en_1m \
	--task_name en_ewt


elif [ "$task" = "nli" ]; then
	## train_joint
	output="en_fgl-j"
	## xlmr ------------------------------------------------
	base_m='xlmr'
	./run_xnli.sh \
	--train_test train_joint \
	--base_model xlmr \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
	--lang_name en


	## mbert ---------------------------------------------
	base_m='mbert'
	./run_xnli.sh \
	--train_test train_joint \
	--base_model mbert \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
	--lang_name en



	## train_lt
	output="en_l"
	## xlmr ------------------------------------------------
	base_m='xlmr'
	./run_xnli.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
	--lang_name en


	## mbert ---------------------------------------------
	base_m='mbert'
	./run_xnli.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
	--lang_name en

	# rm -rf ../experiments/xlmr/${task}/en_ewt/best_model
	# rm -rf ../experiments/xlmr/${task}/en_ewt/checkpoint*
fi

