#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
#SBATCH --output=../tr_output/udp-t.out
#SBATCH --error=../tr_output/udp-t.err
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



task=$1
echo ${task}


source ../vnv/vnv-org/bin/activate


if [ "$task" = "udp" ]; then

	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test train_joint \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --family_path ../adapters/${base_m}/uralic_up_fgl-j/family \
	# --group_path ../adapters/${base_m}/uralic_up_fgl-j/finnic \
	# --lang_path ../adapters/${base_m}/uralic_up_fgl-j/et_1m \
	# --lang_name et_1m \
	# --task_name et_edt


	# ##train-lt

	# #celtic
	# output="ga_idt_l"

	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test train_lt \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --lang_path ../adapters/${base_m}/ie-j/ga \
	# --lang_name ga \
	# --task_name ga_idt

	# #germanic
	# output="en_ewt_l"

	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test train_lt \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --lang_path ../adapters/${base_m}/ie-j/en \
	# --lang_name en \
	# --task_name en_ewt

	# #indic
	# output="mr_ufal_l"

	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test train_lt \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --lang_path ../adapters/${base_m}/ie-j/mr \
	# --lang_name mr \
	# --task_name mr_ufal

	# #iranian
	# output="fa_perdt_l"

	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test train_lt \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --lang_path ../adapters/${base_m}/ie-j/fa \
	# --lang_name fa \
	# --task_name fa_perdt

	# #romance
	# output="es_ancora_l"

	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test train_lt \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --lang_path ../adapters/${base_m}/ie-j/es \
	# --lang_name es \
	# --task_name es_ancora

	# #slavic
	# output="ru_taiga_l"

	# base_m='mbert'
	# ./run_udp.sh \
	# --train_test train_lt \
	# --base_model ${base_m} \
	# --base_dir /scratch/ffaisal/base_models/${base_m} \
	# --data_dir ../data/uralic_demo \
	# --out_dir ../experiments/${base_m}/${task}/${output} \
	# --lang_path ../adapters/${base_m}/ie-j/ru \
	# --lang_name ru \
	# --task_name ru_taiga



	##train-fglt

	#celtic
	output="ga_idt_fgl-j"

	base_m='mbert'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/ie-j/celtic \
 	--group_path ../adapters/${base_m}/ie-j/celtic_goidelic \
	--lang_path ../adapters/${base_m}/ie-j/ga \
	--lang_name ga \
	--task_name ga_idt

	#germanic
	output="en_ewt_fgl-j"

	base_m='mbert'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/ie-j/germanic \
 	--group_path ../adapters/${base_m}/ie-j/germanic_west \
	--lang_path ../adapters/${base_m}/ie-j/en \
	--lang_name en \
	--task_name en_ewt

	#indic
	output="mr_ufal_fgl-j"

	base_m='mbert'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/ie-j/indic \
 	--group_path ../adapters/${base_m}/ie-j/indic_south \
	--lang_path ../adapters/${base_m}/ie-j/mr \
	--lang_name mr \
	--task_name mr_ufal

	#iranian
	output="fa_perdt_fgl-j"

	base_m='mbert'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/ie-j/iranian \
 	--group_path ../adapters/${base_m}/ie-j/iranian_west \
	--lang_path ../adapters/${base_m}/ie-j/fa \
	--lang_name fa \
	--task_name fa_perdt

	#romance
	output="es_ancora_fgl-j"

	base_m='mbert'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/ie-j/romance \
 	--group_path ../adapters/${base_m}/ie-j/romance_west \
	--lang_path ../adapters/${base_m}/ie-j/es \
	--lang_name es \
	--task_name es_ancora

	#slavic
	output="ru_taiga_fgl-j"

	base_m='mbert'
	./run_udp.sh \
	--train_test train_lt \
	--base_model ${base_m} \
	--base_dir /scratch/ffaisal/base_models/${base_m} \
	--data_dir ../data/uralic_demo \
	--out_dir ../experiments/${base_m}/${task}/${output} \
	--family_path ../adapters/${base_m}/ie-j/slavic \
 	--group_path ../adapters/${base_m}/ie-j/slavic_east \
	--lang_path ../adapters/${base_m}/ie-j/ru \
	--lang_name ru \
	--task_name ru_taiga

# 	## germanic ------------------------------------------------
# 	output="en_ewt_fgl-j"
# 	base_m='xlmr'
# 	./run_udp.sh \
# 	--train_test train_joint \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
# 	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
# 	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
# 	--lang_name en_1m \
# 	--task_name en_ewt

# 	output="en_ewt_fgl-j"
# 	base_m='mbert'
# 	./run_udp.sh \
# 	--train_test train_joint \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
# 	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
# 	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
# 	--lang_name en_1m \
# 	--task_name en_ewt

# 	##train-lt
# 	output="en_ewt_l"
# 	base_m='xlmr'
# 	./run_udp.sh \
# 	--train_test train_lt \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
# 	--lang_name en_1m \
# 	--task_name en_ewt

# 	output="en_ewt_l"
# 	base_m='mbert'
# 	./run_udp.sh \
# 	--train_test train_lt \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
# 	--lang_name en_1m \
# 	--task_name en_ewt

# 	# rm -rf ../experiments/xlmr/${task}/en_ewt/best_model
# 	# rm -rf ../experiments/xlmr/${task}/en_ewt/checkpoint*

# elif [ "$task" = "pos" ]; then
# 	## uralic -------------------------------------------------
# 	##train_joint----xlmr
# 	output="et_edt_fgl-j"
# 	base_m='xlmr'
# 	./run_ner.sh \
# 	--train_test train_joint \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--family_path ../adapters/${base_m}/uralic_up_fgl-j/family \
# 	--group_path ../adapters/${base_m}/uralic_up_fgl-j/finnic \
# 	--lang_path ../adapters/${base_m}/uralic_up_fgl-j/et_1m \
# 	--lang_name et_1m \
# 	--task_name et_edt

# 	##train_joint----mbert
# 	base_m='mbert'
# 	./run_ner.sh \
# 	--train_test train_joint \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--family_path ../adapters/${base_m}/uralic_up_fgl-j/family \
# 	--group_path ../adapters/${base_m}/uralic_up_fgl-j/finnic \
# 	--lang_path ../adapters/${base_m}/uralic_up_fgl-j/et_1m \
# 	--lang_name et_1m \
# 	--task_name et_edt

# 	output="et_edt_l"
# 	##train_lt----xlmr
# 	base_m='xlmr'
# 	./run_ner.sh \
# 	--train_test train_lt \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--lang_path ../adapters/${base_m}/uralic_up_l/et_1m/mlm \
# 	--lang_name et_1m \
# 	--task_name et_edt

# 	##train_lt----mbert
# 	base_m='mbert'
# 	./run_ner.sh \
# 	--train_test train_lt \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--lang_path ../adapters/${base_m}/uralic_up_l/et_1m/mlm \
# 	--lang_name et_1m \
# 	--task_name et_edt



# 	## germanic ------------------------------------------------
# 	output="en_ewt_fgl-j"
# 	#joint
# 	base_m='xlmr'
# 	./run_ner.sh \
# 	--train_test train_joint \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
# 	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
# 	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
# 	--lang_name en_1m \
# 	--task_name en_ewt

# 	base_m='mbert'
# 	./run_ner.sh \
# 	--train_test train_joint \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
# 	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
# 	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
# 	--lang_name en_1m \
# 	--task_name en_ewt

# 	#lt
# 	output="en_ewt_l"
# 	base_m='xlmr'
# 	./run_ner.sh \
# 	--train_test train_lt \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
# 	--lang_name en_1m \
# 	--task_name en_ewt

# 	base_m='mbert'
# 	./run_ner.sh \
# 	--train_test train_lt \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
# 	--lang_name en_1m \
# 	--task_name en_ewt


# elif [ "$task" = "nli" ]; then
# 	## train_joint
# 	output="en_fgl-j"
# 	## xlmr ------------------------------------------------
# 	base_m='xlmr'
# 	./run_xnli.sh \
# 	--train_test train_joint \
# 	--base_model xlmr \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
# 	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
# 	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
# 	--lang_name en


# 	## mbert ---------------------------------------------
# 	base_m='mbert'
# 	./run_xnli.sh \
# 	--train_test train_joint \
# 	--base_model mbert \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--family_path ../adapters/${base_m}/germanic_up_fgl-j/family \
# 	--group_path ../adapters/${base_m}/germanic_up_fgl-j/west \
# 	--lang_path ../adapters/${base_m}/germanic_up_fgl-j/en_1m \
# 	--lang_name en



# 	## train_lt
# 	output="en_l"
# 	## xlmr ------------------------------------------------
# 	base_m='xlmr'
# 	./run_xnli.sh \
# 	--train_test train_lt \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
# 	--lang_name en


# 	## mbert ---------------------------------------------
# 	base_m='mbert'
# 	./run_xnli.sh \
# 	--train_test train_lt \
# 	--base_model ${base_m} \
# 	--base_dir /scratch/ffaisal/base_models/${base_m} \
# 	--data_dir ../data/uralic_demo \
# 	--out_dir ../experiments/${base_m}/${task}/${output} \
# 	--lang_path ../adapters/${base_m}/germanic_up_l/en_1m/mlm \
# 	--lang_name en

# 	# rm -rf ../experiments/xlmr/${task}/en_ewt/best_model
# 	# rm -rf ../experiments/xlmr/${task}/en_ewt/checkpoint*
fi

