dataset="universal_dependencies"
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
task_path_j=${task_path:-en_ewt}
cache_dir=${cache_dir:-/scratch/ffaisal/hug_cache}
num_epoch=${num_epoch:-3}


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




if [ "$train_test" = "train_joint" ]; then
    ###train:lang+region+family (joint)
    echo ${lang_path}
    echo ${family_path}
    echo ${group_path}
    echo ${out_dir}
    python ../examples/token-classification/run_ner_all.py \
        --model_name_or_path ${base_dir} \
        --do_train \
        --task_name pos \
        --dataset_name $dataset \
        --load_lang_adapter ${lang_path} \
        --lang_adapter_config ${lang_path}/adapter_config.json \
        --load_family_adapter ${family_path} \
        --family_adapter_config ${family_path}/adapter_config.json \
        --load_region_adapter ${group_path} \
        --region_adapter_config ${group_path}/adapter_config.json \
        --language ${lang_name} \
        --dataset_config_name ${task_name} \
        --label_column_name upos \
        --per_device_train_batch_size ${train_batch} \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${num_epoch} \
        --max_seq_length ${seq_length} \
        --output_dir ${out_dir} \
        --overwrite_output_dir \
        --train_adapter \
        --save_steps 5000 \
        --cache_dir ${cache_dir} \
        --overwrite_output_dir

elif [ "$train_test" = "train_lt" ]; then
    ###train:lang+region+family (joint)
    echo ${lang_path}
    echo ${family_path}
    echo ${group_path}
    echo ${out_dir}
    python ../examples/token-classification/run_ner_all.py \
        --model_name_or_path ${base_dir} \
        --do_train \
        --task_name pos \
        --dataset_name $dataset \
        --load_lang_adapter ${lang_path} \
        --lang_adapter_config ${lang_path}/adapter_config.json \
        --language ${lang_name} \
        --dataset_config_name ${task_name} \
        --label_column_name upos \
        --per_device_train_batch_size ${train_batch} \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${num_epoch} \
        --max_seq_length ${seq_length} \
        --output_dir ${out_dir} \
        --overwrite_output_dir \
        --train_adapter \
        --save_steps 5000 \
        --cache_dir ${cache_dir} \
        --overwrite_output_dir

elif [ "$train_test" = "predict_all" ]; then
    # export TASK_NAME="en_ewt"
    python ../new_scripts/run_ner_all.py \
        --model_name_or_path ${base_dir} \
        --task_name ${task_name} \
        --task_path ${task_path} \
        --task_path_j ${task_path_j} \
        --dataset_name $dataset \
        --dataset_config_name ${task_name} \
        --label_column_name upos \
        --lang_config ${lang_config} \
        --do_predict_all \
        --family_name ${family_name} \
        --per_device_train_batch_size ${train_batch} \
        --learning_rate ${learning_rate} \
        --max_seq_length ${seq_length} \
        --output_dir ${out_dir} \
        --overwrite_output_dir \
        --evaluation_strategy epoch \
        --cache_dir ${cache_dir} \
        --overwrite_output_dir

fi

# ##train:lang+family (joint lang+family+region)
# folder="${domain}_lang_region_joint"
# lang_path="../adapter/${folder}/${lang_file}"
# family_path="../adapter/${folder}/family"
# output_dir_n=${TASK_NAME}_joint_lang_family
# echo ${lang_path}
# echo ${family_path}
# echo ${output_dir_n}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#     --model_name_or_path $model_path \
#     --do_train \
#     --task_name pos \
#     --dataset_name $dataset \
#     --load_lang_adapter ${lang_path} \
#     --lang_adapter_config ${lang_path}/adapter_config.json \
#     --load_family_adapter ${family_path} \
#     --family_adapter_config ${family_path}/adapter_config.json \
#     --language ${lang_file} \
#     --dataset_config_name $TASK_NAME \
#     --label_column_name upos \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3 \
#     --output_dir ../experiments/${task}/${output_dir_n} \
#     --overwrite_output_dir \
#     --save_steps 5000 \
#     --max_seq_length 256 \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --overwrite_output_dir


# ###train:lang (not joint)
# folder="${domain}_lang"
# lang_path="../adapter/${folder}/${lang_file}/mlm"
# output_dir_n=${TASK_NAME}_lang
# echo ${lang_path}
# echo ${family_path}
# echo ${output_dir_n}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#     --model_name_or_path $model_path \
#     --do_train \
#     --task_name pos \
#     --dataset_name $dataset \
#     --load_lang_adapter ${lang_path} \
#     --lang_adapter_config ${lang_path}/adapter_config.json \
#     --language ${lang_file} \
#     --dataset_config_name $TASK_NAME \
#     --label_column_name upos \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3 \
#     --output_dir ../experiments/${task}/${output_dir_n} \
#     --overwrite_output_dir \
#     --save_steps 5000 \
#     --max_seq_length 256 \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --overwrite_output_dir

# ###train:task
# output_dir_n=${TASK_NAME}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#   --model_name_or_path $model_path \
#   --dataset_name $dataset \
#   --dataset_config_name $TASK_NAME \
#   --label_column_name upos \
#   --per_device_train_batch_size 16 \
#   --learning_rate 5e-4 \
#   --num_train_epochs 3 \
#   --output_dir ../experiments/${task}/${output_dir_n} \
#   --do_train \
#   --task_name pos \
#   --save_steps 5000 \
#   --max_seq_length 256 \
#   --train_adapter \
#   --cache_dir /scratch/ffaisal/hug_cache \
#   --overwrite_output_dir

#######################-----------------------------------------------------------------------------------


###train:lang+region+family (joint)
# folder="${domain}_lang_region_joint"
# lang_path="../adapter/${folder}/${lang_file}"
# family_path="../adapter/${folder}/family"
# region_path="../adapter/${folder}/${lang_reg}"
# output_dir_n=${TASK_NAME}_joint_lang_family_region
# echo ${lang_path}
# echo ${family_path}
# echo ${region_path}
# echo ${output_dir_n}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#     --model_name_or_path $model_path \
#     --do_train \
#     --task_name pos \
#     --dataset_name $dataset \
#     --load_lang_adapter ${lang_path} \
#     --lang_adapter_config ${lang_path}/adapter_config.json \
#     --load_family_adapter ${family_path} \
#     --family_adapter_config ${family_path}/adapter_config.json \
#     --load_region_adapter ${region_path} \
#     --region_adapter_config ${region_path}/adapter_config.json \
#     --language ${lang_file} \
#     --dataset_config_name $TASK_NAME \
#     --label_column_name upos \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3 \
#     --output_dir ../experiments/${task}/${output_dir_n} \
#     --overwrite_output_dir \
#     --max_seq_length 256 \
#     --train_adapter \
#     --save_steps 5000 \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --overwrite_output_dir


# ##train:lang+family (joint lang+family+region)
# folder="${domain}_lang_region_joint"
# lang_path="../adapter/${folder}/${lang_file}"
# family_path="../adapter/${folder}/family"
# output_dir_n=${TASK_NAME}_joint_lang_family
# echo ${lang_path}
# echo ${family_path}
# echo ${output_dir_n}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#     --model_name_or_path $model_path \
#     --do_train \
#     --task_name pos \
#     --dataset_name $dataset \
#     --load_lang_adapter ${lang_path} \
#     --lang_adapter_config ${lang_path}/adapter_config.json \
#     --load_family_adapter ${family_path} \
#     --family_adapter_config ${family_path}/adapter_config.json \
#     --language ${lang_file} \
#     --dataset_config_name $TASK_NAME \
#     --label_column_name upos \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3 \
#     --output_dir ../experiments/${task}/${output_dir_n} \
#     --overwrite_output_dir \
#     --save_steps 5000 \
#     --max_seq_length 256 \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --overwrite_output_dir

# ##train:lang+family (joint lang+family)
# folder="${domain}_lang_joint"
# lang_path="../adapter/${folder}/${lang_file}"
# family_path="../adapter/${folder}/family"
# output_dir_n=${TASK_NAME}_joint_lang_family_ntrg
# echo ${lang_path}
# echo ${family_path}
# echo ${output_dir_n}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#     --model_name_or_path $model_path \
#     --do_train \
#     --task_name pos \
#     --dataset_name $dataset \
#     --load_lang_adapter ${lang_path} \
#     --lang_adapter_config ${lang_path}/adapter_config.json \
#     --load_family_adapter ${family_path} \
#     --family_adapter_config ${family_path}/adapter_config.json \
#     --language ${lang_file} \
#     --dataset_config_name $TASK_NAME \
#     --label_column_name upos \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3 \
#     --output_dir ../experiments/${task}/${output_dir_n} \
#     --overwrite_output_dir \
#     --save_steps 5000 \
#     --max_seq_length 256 \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --overwrite_output_dir


# ###train:lang (joint lang+family+region)
# folder="${domain}_lang_region_joint"
# lang_path="../adapter/${folder}/${lang_file}"
# echo ${lang_path}
# output_dir_n=${TASK_NAME}_joint_lang
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#     --model_name_or_path $model_path \
#     --do_train \
#     --task_name pos \
#     --dataset_name $dataset \
#     --load_lang_adapter ${lang_path} \
#     --lang_adapter_config ${lang_path}/adapter_config.json \
#     --language ${lang_file} \
#     --dataset_config_name $TASK_NAME \
#     --label_column_name upos \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3 \
#     --output_dir ../experiments/${task}/${output_dir_n} \
#     --overwrite_output_dir \
#     --save_steps 5000 \
#     --max_seq_length 256 \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --overwrite_output_dir

# ###train:lang+family (not joint)
# folder="${domain}_lang"
# lang_path="../adapter/${folder}/${lang_file}/mlm"
# family_path="../adapter/${folder}/family/mlm"
# output_dir_n=${TASK_NAME}_lang_family
# echo ${lang_path}
# echo ${family_path}
# echo ${output_dir_n}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#     --model_name_or_path $model_path \
#     --do_train \
#     --task_name pos \
#     --dataset_name $dataset \
#     --load_lang_adapter ${lang_path} \
#     --lang_adapter_config ${lang_path}/adapter_config.json \
#     --load_family_adapter ${family_path} \
#     --family_adapter_config ${family_path}/adapter_config.json \
#     --language ${lang_file} \
#     --dataset_config_name $TASK_NAME \
#     --label_column_name upos \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3 \
#     --output_dir ../experiments/${task}/${output_dir_n} \
#     --overwrite_output_dir \
#     --save_steps 5000 \
#     --max_seq_length 256 \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --overwrite_output_dir


# ###train:lang (not joint)
# folder="${domain}_lang"
# lang_path="../adapter/${folder}/${lang_file}/mlm"
# output_dir_n=${TASK_NAME}_lang
# echo ${lang_path}
# echo ${family_path}
# echo ${output_dir_n}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#     --model_name_or_path $model_path \
#     --do_train \
#     --task_name pos \
#     --dataset_name $dataset \
#     --load_lang_adapter ${lang_path} \
#     --lang_adapter_config ${lang_path}/adapter_config.json \
#     --language ${lang_file} \
#     --dataset_config_name $TASK_NAME \
#     --label_column_name upos \
#     --per_device_train_batch_size 16 \
#     --learning_rate 5e-4 \
#     --num_train_epochs 3 \
#     --output_dir ../experiments/${task}/${output_dir_n} \
#     --overwrite_output_dir \
#     --save_steps 5000 \
#     --max_seq_length 256 \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --overwrite_output_dir

# ###train:task
# output_dir_n=${TASK_NAME}
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#   --model_name_or_path $model_path \
#   --dataset_name $dataset \
#   --dataset_config_name $TASK_NAME \
#   --label_column_name upos \
#   --per_device_train_batch_size 16 \
#   --learning_rate 5e-4 \
#   --num_train_epochs 3 \
#   --output_dir ../experiments/${task}/${output_dir_n} \
#   --do_train \
#   --task_name pos \
#   --save_steps 5000 \
#   --max_seq_length 256 \
#   --train_adapter \
#   --cache_dir /scratch/ffaisal/hug_cache \
#   --overwrite_output_dir


##predict:predic all
# output_dir_n='results_uralic_up'
# python ../adapter-transformers/examples/token-classification/run_ner_all.py \
#   --model_name_or_path $model_path \
#   --dataset_name $dataset \
#   --dataset_config_name $TASK_NAME \
#   --label_column_name upos \
#   --per_device_train_batch_size 16 \
#   --learning_rate 5e-4 \
#   --num_train_epochs 3 \
#   --output_dir ../experiments/${task}/${output_dir_n} \
#   --do_predict_all \
#   --task_name pos \
#   --save_steps 5000 \
#   --max_seq_length 256 \
#   --cache_dir /scratch/ffaisal/hug_cache
