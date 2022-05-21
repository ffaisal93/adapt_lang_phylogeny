#!/bin/bash


# ./run_mlm_all.sh \
# --train_test train \
# --base_model xlmr \
# --base_dir /scratch/ffaisal/base_models/xlmr \
# --data_dir ../data/uralic_demo \
# --out_dir ../adapters/uralic_demo \
# --lang_family uralic 


train_test=${train_test:-train}
base_model=${base_model:-mbert}
base_dir=${base_dir:-/scratch/ffaisal/base_models/pytorch_mbert}
data_dir=${data_dir:-../data/uralic_demo}
out_dir=${out_dir:-../adapters/uralic_demo}
lang_config=${lang_config:-../meta_files/lang_meta.json}
lang_family=${lang_family:-uralic}
cache_dir=${cache_dir:-/scratch/ffaisal/hug_cache}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2 #Optional to see the parameter:value result
   fi

  shift
done



if [ "$base_model" = "mbert" ]; then
    seq_length=512
    learning_rate=1e-4
    train_batch=12
elif [ "$base_model" = "xlmr" ]; then
    seq_length=256
    learning_rate=2e-5
    train_batch=8
fi


if [ "$train_test" = "train_joint" ]; then
    ######train with region---[americas nli mbert]-------------------------

    python ../examples/language-modeling/run_mlm_with_region.py \
        --model_name_or_path  ${base_dir} \
        --train_files ${data_dir} \
        --lang_config ${lang_config} \
        --lang_family ${lang_family} \
        --do_train \
        --learning_rate ${learning_rate} \
        --num_train_epochs 40 \
        --output_dir ${out_dir} \
        --train_adapter \
        --cache_dir ${cache_dir} \
        --adapter_config "pfeiffer+inv" \
        --max_seq_length ${seq_length} \
        --save_steps 5000 \
        --per_device_train_batch_size ${train_batch} \
        --adapter_reduction_factor 16 \
        --overwrite_output_dir
fi


######train without region----------------------------
# python ../test_adapt/adapter-transformers/examples/language-modeling/run_mlm_all.py \
#     --model_name_or_path  /scratch/ffaisal/base_models/pytorch_mbert \
#     --train_file ../mono-data/af1.txt \
#     --train_files ../uralic \
#     --do_train \
#     --learning_rate 1e-4 \
#     --num_train_epochs 3.0 \
#     --output_dir ../adapter/ura_lang_joint \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --adapter_config "pfeiffer+inv" \
#     --max_seq_length 512 \
#     --save_steps 5000 \
#     --per_device_train_batch_size 12


# python ../test_adapt/adapter-transformers/examples/language-modeling/run_mlm_all.py \
#     --model_name_or_path  /scratch/ffaisal/base_models/pytorch_mbert \
#     --train_file ../mono-data/af1.txt \
#     --train_files ../1m_data \
#     --do_train \
#     --learning_rate 1e-4 \
#     --num_train_epochs 3.0 \
#     --output_dir ../adapter/1m_lang_joint \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --adapter_config "pfeiffer+inv" \
#     --max_seq_length 512 \
#     --per_device_train_batch_size 12

# ######train with region----------------------------

# python ../test_adapt/adapter-transformers/examples/language-modeling/run_mlm_with_region.py \
#     --model_name_or_path  /scratch/ffaisal/base_models/xlmr \
#     --train_file ../mono-data/af1.txt \
#     --train_files ../americasnlp_lang/uto_aztecan \
#     --lang_config ../lang_meta.json \
#     --lang_family uto_aztecan \
#     --do_train \
#     --learning_rate 2e-5 \
#     --num_train_epochs 40.0 \
#     --output_dir ../adapter/uto_aztecan_40 \
#     --train_adapter \
#     --cache_dir /scratch/ffaisal/hug_cache \
#     --adapter_config "pfeiffer+inv" \
#     --max_seq_length 256 \
#     --save_steps 5000 \
#     --per_device_train_batch_size 8 \
#     --adapter_reduction_factor 16 \
#     --overwrite_output_dir



