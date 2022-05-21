#!/bin/bash
task=$1
echo ${task}
source ../vnv/vnv-org/bin/activate


if [ "$task" = "udp" ]; then
	# ##---------------------------------------------------------
	# family_name='uralic_up'
	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test predict_all \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --lang_config ../meta_files/pred_meta_${base_m}.json \
	# --family_name ${family_name} \
	# --task_name et_edt \
	# --out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	# --task_path ../experiments/${base_m}/${task}/et_edt_l/ud_et_edt \
	# --task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/ud_et_edt

	# base_m='xlmr'
	# ./run_udp.sh \
	# --train_test predict_all \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --lang_config ../meta_files/pred_meta_${base_m}.json \
	# --family_name ${family_name} \
	# --task_name et_edt \
	# --out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	# --task_path ../experiments/${base_m}/${task}/et_edt_l/ud_et_edt \
	# --task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/ud_et_edt

	# ##---------------------------------------------------------
	# family_name='germanic_up'
	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test predict_all \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --lang_config ../meta_files/pred_meta_${base_m}.json \
	# --family_name ${family_name} \
	# --task_name et_edt \
	# --out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	# --task_path ../experiments/${base_m}/${task}/en_ewt_l/ud_en_ewt \
	# --task_path_j ../experiments/${base_m}/${task}/en_ewt_fgl-j/ud_en_ewt

	# base_m='xlmr'
	# ./run_udp.sh \
	# --train_test predict_all \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --lang_config ../meta_files/pred_meta_${base_m}.json \
	# --family_name ${family_name} \
	# --task_name et_edt \
	# --out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	# --task_path ../experiments/${base_m}/${task}/en_ewt_l/ud_en_ewt \
	# --task_path_j ../experiments/${base_m}/${task}/en_ewt_fgl-j/ud_en_ewt

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
	--task_path ../experiments/${base_m}/${task}/en_ewt_l/ud_en_ewt \
	--task_path_j ../experiments/${base_m}/${task}/en_ewt_fgl-j/ud_en_ewt

	base_m='xlmr'
	./run_udp.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/en_ewt_l/ud_en_ewt \
	--task_path_j ../experiments/${base_m}/${task}/en_ewt_fgl-j/ud_en_ewt

elif [ "$task" = "pos" ]; then
	##---------------------------------------------------------
	family_name='uralic_up'
	base_m='mbert'
	./run_ner.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_l/pos_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/pos_et_edt

	base_m='xlmr'
	./run_ner.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name et_edt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/et_edt_l/pos_et_edt \
	--task_path_j ../experiments/${base_m}/${task}/et_edt_fgl-j/pos_et_edt

	# ##---------------------------------------------------------
	family_name='germanic_up'
	base_m='mbert'
	./run_ner.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name en_ewt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/en_ewt_l/pos_en_ewt \
	--task_path_j ../experiments/${base_m}/${task}/en_ewt_fgl-j/pos_en_ewt

	base_m='xlmr'
	./run_ner.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name en_ewt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/en_ewt_l/pos_en_ewt \
	--task_path_j ../experiments/${base_m}/${task}/en_ewt_fgl-j/pos_en_ewt

	# ##---------------------------------------------------------
	family_name='tupian'
	base_m='mbert'
	./run_ner.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name en_ewt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/en_ewt_l/pos_en_ewt \
	--task_path_j ../experiments/${base_m}/${task}/en_ewt_fgl-j/pos_en_ewt

	base_m='xlmr'
	./run_ner.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--task_name en_ewt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/en_ewt_l/pos_en_ewt \
	--task_path_j ../experiments/${base_m}/${task}/en_ewt_fgl-j/pos_en_ewt

elif [ "$task" = "nli" ]; then
	## train_joint
	# family_name="tupian"
	# ## xlmr ------------------------------------------------
	# base_m='xlmr'
	# ./run_xnli.sh \
	# --train_test predict_all \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --lang_config ../meta_files/pred_meta_${base_m}.json \
	# --family_name ${family_name} \
	# --dataset americas_nli \
	# --task_name en_ewt \
	# --out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	# --task_path ../experiments/${base_m}/${task}/en_l/xnli_en \
	# --task_path_j ../experiments/${base_m}/${task}/en_fgl-j/xnli_en \
	# --lang_name en

	family_name="tupian"
	## xlmr ------------------------------------------------
	base_m='mbert'
	./run_xnli.sh \
	--train_test predict_all \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--lang_config ../meta_files/pred_meta_${base_m}.json \
	--family_name ${family_name} \
	--dataset americas_nli \
	--task_name en_ewt \
	--out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	--task_path ../experiments/${base_m}/${task}/en_l/xnli_en \
	--task_path_j ../experiments/${base_m}/${task}/en_fgl-j/xnli_en \
	--lang_name en


	# family_name="uto_aztecan"
	# ## xlmr ------------------------------------------------
	# base_m='xlmr'
	# ./run_xnli.sh \
	# --train_test predict_all \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --lang_config ../meta_files/pred_meta_${base_m}.json \
	# --family_name ${family_name} \
	# --dataset americas_nli \
	# --task_name en_ewt \
	# --out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	# --task_path ../experiments/${base_m}/${task}/en_l/xnli_en \
	# --task_path_j ../experiments/${base_m}/${task}/en_fgl-j/xnli_en \
	# --lang_name en

	# family_name="uto_aztecan"
	# ## xlmr ------------------------------------------------
	# base_m='mbert'
	# ./run_xnli.sh \
	# --train_test predict_all \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --lang_config ../meta_files/pred_meta_${base_m}.json \
	# --family_name ${family_name} \
	# --dataset americas_nli \
	# --task_name en_ewt \
	# --out_dir ../experiments/${base_m}/${task}/result/${family_name} \
	# --task_path ../experiments/${base_m}/${task}/en_l/xnli_en \
	# --task_path_j ../experiments/${base_m}/${task}/en_fgl-j/xnli_en \
	# --lang_name en

fi
