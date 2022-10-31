#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
##SBATCH --output=../tr_output/pos.out
##SBATCH --error=../tr_output/pos.err
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


if [ "$task" = "udp_random" ]; then
	##---------------------------------------------------------
	family_name='random-j'
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_mh} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name en_ewt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/udp/en_ewt_l/ud_en_ewt \
	--task_path_j ../experiments/${base_m}/udp/en_ewt_fgl-j/ud_en_ewt

elif [ "$task" = "pos_random" ]; then
	## uralic -------------------------------------------------
	##train_joint----xlmr
	family_name='random-j'
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'

	./run_ner.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_mh} \
	--data_dir ../data/uralic_demo \
	--family_name ${family_name} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/pos/en_ewt_l/pos_en_ewt \
	--task_path_j ../experiments/${base_m}/pos/en_ewt_fgl-j/pos_en_ewt \
	--task_name en_ewt



if [ "$task" = "udp_lll" ]; then
	##---------------------------------------------------------
	family_name='uralic_lll'
	base_m='mbert'
	base_mh='bert-base-multilingual-cased'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_mh} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_lll-j/ud_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_lll-j/ud_et_edt



if [ "$task" = "udp" ]; then
	##---------------------------------------------------------
	family_name='uralic_up'
	base_m='mbert'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_l/ud_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/ud_et_edt

	base_m='xlmr'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_l/ud_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/ud_et_edt

	##---------------------------------------------------------
	family_name='germanic_up'
	base_m='mbert'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_l/ud_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/ud_et_edt

	base_m='xlmr'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_l/ud_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/ud_et_edt

	##---------------------------------------------------------
	family_name='tupian'
	base_m='mbert'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_l/ud_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/ud_et_edt

	base_m='xlmr'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_l/ud_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/ud_et_edt






	# rm -rf ../experiments/xlmr/${task}/en_ewt/best_model
	# rm -rf ../experiments/xlmr/${task}/en_ewt/checkpoint*

elif [ "$task" = "pos" ]; then
	## uralic -------------------------------------------------
	##train_joint----xlmr
	output="et_edt_fgl-j"
	base_m='xlmr'
	./run_ner.sh \
	--train_test predict_all \
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
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/uralic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/uralic_up_fgl-j/finnic \
	--lang_path ../adapters/${base_m}/uralic_up_fgl-j/et_1m \
	--lang_name et_1m \
	--task_name et_edt

elif [ "$task" = "nli" ]; then
	## train_joint
	output="en_fgl-j"
	## xlmr ------------------------------------------------
	base_m='xlmr'
	./run_xnli.sh \
	--train_test predict_all \
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
	--train_test predict_all \
	--base_model mbert \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
	--lang_name en

	# rm -rf ../experiments/xlmr/${task}/en_ewt/best_model
	# rm -rf ../experiments/xlmr/${task}/en_ewt/checkpoint*
fi

