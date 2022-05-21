source /projects/antonis/fahim/venvs/hopper/adapt/bin/activate

# domain="ama"
# lang_file="en_1m"
# lang_reg="west"
# TASK_NAME="xnli"
# task="xnli"
# # dataset="xnli"
# dataset="americas_nli"
# train_lang="en"
# model_path="/scratch/ffaisal/base_models/xlmr"

# domain="ger"
# lang_file="en_1m"
# lang_reg="north"
# TASK_NAME="en_ewt"
# task="pos"

dataset=${dataset:-xnli}
train_test=${train_test:-train_joint}
base_model=${base_model:-mbert}
base_dir=${base_dir:-/scratch/ffaisal/base_models/pytorch_mbert}
out_dir=${out_dir:-../adapters/uralic_demo}
lang_config=${lang_config:-../meta_files/lang_meta.json}
family_name=${family_name:-../meta_files/pred_meta.json}
family_path=${family_path:-uralic}
group_path=${group_path:-uralic}
lang_path=${lang_path:-uralic}
lang_name=${lang_name:-uralic}
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
    seq_length=128
    learning_rate=2e-5
    train_batch=16
elif [ "$base_model" = "xlmr" ]; then
    seq_length=128
    learning_rate=2e-5
    train_batch=16
fi


if [ "$train_test" = "train_joint" ]; then
    ###train:lang+region+family (joint)
    echo ${lang_path}
    echo ${family_path}
    echo ${group_path}
    echo ${out_dir}
    python ../examples/text-classification/run_xnli.py \
        --model_name_or_path ${base_dir} \
        --language ${lang_name} \
        --do_train \
        --task_name xnli \
        --dataset_name ${dataset} \
        --load_lang_adapter ${lang_path} \
        --lang_adapter_config ${lang_path}/adapter_config.json \
        --load_family_adapter ${family_path} \
        --family_adapter_config ${family_path}/adapter_config.json \
        --load_region_adapter ${group_path} \
        --region_adapter_config ${group_path}/adapter_config.json \
        --train_language ${lang_name} \
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
    python ../examples/text-classification/run_xnli.py \
        --model_name_or_path ${base_dir} \
        --language ${lang_name} \
        --do_train \
        --task_name xnli \
        --dataset_name ${dataset} \
        --load_lang_adapter ${lang_path} \
        --lang_adapter_config ${lang_path}/adapter_config.json \
        --train_language ${lang_name} \
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
    ###train:lang+region+family (joint)
    echo ${lang_path}
    echo ${family_path}
    echo ${group_path}
    echo ${out_dir}
    python ../new_scripts/run_xnli.py \
        --model_name_or_path ${base_dir} \
        --language ${lang_name} \
        --task_name xnli \
        --task_path ${task_path} \
        --task_path_j ${task_path_j} \
        --lang_config ${lang_config} \
        --dataset_name ${dataset} \
        --family_name ${family_name} \
        --train_language ${lang_name} \
        --test_language aym \
        --per_device_train_batch_size ${train_batch} \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${num_epoch} \
        --max_seq_length ${seq_length} \
        --output_dir ${out_dir} \
        --overwrite_output_dir \
        --do_predict_all \
        --save_steps 5000 \
        --cache_dir ${cache_dir} \
        --overwrite_output_dir

    # python ../new_scripts/run_ner_all.py \
    #     --model_name_or_path ${base_dir} \
    #     --task_name ${task_name} \
    #     --task_path ${task_path} \
    #     --task_path_j ${task_path_j} \
    #     --dataset_name $dataset \
    #     --dataset_config_name ${task_name} \
    #     --label_column_name upos \
    #     --lang_config ${lang_config} \
    #     --do_predict_all \
    #     --family_name ${family_name} \
    #     --lang_name aym \
    #     --per_device_train_batch_size ${train_batch} \
    #     --learning_rate ${learning_rate} \
    #     --max_seq_length ${seq_length} \
    #     --output_dir ${out_dir} \
    #     --cache_dir ${cache_dir} \
    #     --overwrite_output_dir

fi