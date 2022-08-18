#!/bin/bash


# ./run_udp.sh \
# --train_test train_joint \
# --base_model xlmr \
# --base_dir /scratch/ffaisal/base_models/xlmr \
# --data_dir ../data/uralic_demo \
# --out_dir ../experiments/udp/uralic_fgl-j \
# --family_path ../adapters/xlmr/uralic_up_fgl-j/family \
# --group_path ../adapters/xlmr/uralic_up_fgl-j/finnic \
# --lang_path ../adapters/xlmr/uralic_up_fgl-j/et_1m \
# --task_path ../adapters/xlmr/uralic_up_fgl-j/et_1m \
# --lang_name et_1m \
# --task_name et_edt


train_test=${train_test:-train_joint}
base_model=${base_model:-mbert}
base_dir=${base_dir:-/scratch/ffaisal/base_models/pytorch_mbert}
data_dir=${data_dir:-../data/uralic_demo}
out_dir=${out_dir:-../adapters/uralic_demo}
lang_config=${lang_config:-../meta_files/lang_meta.json}
family_name=${family_name:-../meta_files/pred_meta.json}
family_path=${family_path:-uralic}
group_path=${group_path:-uralic}
lang_path=${lang_path:-uralic}
lang_name=${lang_name:-uralic}
task_name=${task_name:-en_ewt}
task_path=${task_path:-en_ewt}
task_path_j=${task_path_j:-en_ewt}
cache_dir=${cache_dir:-/scratch/ffaisal/hug_cache}
num_epoch=${num_epoch:-6}


while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2 #Optional to see the parameter:value result
   fi

  shift
done



if [ "$base_model" = "mbert" ]; then
    seq_length=256
    learning_rate=5e-4
    train_batch=16
elif [ "$base_model" = "xlmr" ]; then
    seq_length=256
    learning_rate=5e-4
    train_batch=16
fi




model_path="/scratch/ffaisal/base_models/xlmr"



if [ "$train_test" = "train_joint" ]; then
    ###train:lang+region+family (joint)
    echo ${lang_path}
    echo ${family_path}
    echo ${region_path}
    echo ${out_dir}
    python ../examples/dependency-parsing/run_udp_america.py \
        --model_name_or_path ${base_dir} \
        --do_train \
        --load_lang_adapter ${lang_path} \
        --lang_adapter_config ${lang_path}/adapter_config.json \
        --load_family_adapter ${family_path} \
        --family_adapter_config ${family_path}/adapter_config.json \
        --load_region_adapter ${group_path} \
        --region_adapter_config ${group_path}/adapter_config.json \
        --language ${lang_name} \
        --task_name ${task_name} \
        --per_device_train_batch_size ${train_batch} \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${num_epoch} \
        --max_seq_length ${seq_length} \
        --output_dir ${out_dir} \
        --overwrite_output_dir \
        --store_best_model \
        --evaluation_strategy epoch \
        --metric_score uas \
        --train_adapter \
        --cache_dir ${cache_dir} \
        --overwrite_output_dir

elif [ "$train_test" = "train_lt" ]; then
    ###train:lang+region+family (joint)
    echo ${lang_path}
    echo ${out_dir}
    python ../examples/dependency-parsing/run_udp_america.py \
        --model_name_or_path ${base_dir} \
        --do_train \
        --load_lang_adapter ${lang_path} \
        --lang_adapter_config ${lang_path}/adapter_config.json \
        --language ${lang_name} \
        --task_name ${task_name} \
        --per_device_train_batch_size ${train_batch} \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${num_epoch} \
        --max_seq_length ${seq_length} \
        --output_dir ${out_dir} \
        --overwrite_output_dir \
        --store_best_model \
        --evaluation_strategy epoch \
        --metric_score uas \
        --train_adapter \
        --cache_dir ${cache_dir} \
        --overwrite_output_dir

elif [ "$train_test" = "predict_all_ie" ]; then
    # export TASK_NAME="en_ewt"
    python ../examples/dependency-parsing/run_udp_ie.py \
        --model_name_or_path ${base_dir} \
        --task_name ${task_name} \
        --task_path ${task_path} \
        --task_path_j ${task_path_j} \
        --lang_config ${lang_config} \
        --do_predict_all \
        --family_name ${family_name} \
        --per_device_train_batch_size ${train_batch} \
        --learning_rate ${learning_rate} \
        --max_seq_length ${seq_length} \
        --output_dir ${out_dir} \
        --overwrite_output_dir \
        --store_best_model \
        --evaluation_strategy epoch \
        --metric_score uas \
        --cache_dir ${cache_dir} \
        --overwrite_output_dir

fi


