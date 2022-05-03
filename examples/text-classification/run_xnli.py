#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers

import transformers.adapters.composition as ac
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "xnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),

}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # language: str = field(
    #     default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    # )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."}
    )
    test_language: Optional[str] = field(
        default=None, metadata={"help": "Test language if it is different from the evaluation language."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if training_args.do_train:
        if model_args.train_language is None:
            train_dataset = load_dataset(data_args.dataset_name, model_args.language, split="train", cache_dir=model_args.cache_dir)
        else:
            train_dataset = load_dataset(
                data_args.dataset_name, model_args.train_language, split="train", cache_dir=model_args.cache_dir
            )
        label_list = train_dataset.features["label"].names

    if training_args.do_eval:
        eval_dataset = load_dataset(data_args.dataset_name, model_args.language, split="validation", cache_dir=model_args.cache_dir)
        label_list = eval_dataset.features["label"].names

    if training_args.do_predict or training_args.do_predict_all:
        predict_dataset = load_dataset(data_args.dataset_name, model_args.test_language, split="test", cache_dir=model_args.cache_dir)
        label_list = predict_dataset.features["label"].names

    # Labels
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="xnli",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    task_name = data_args.task_name +"_" + model_args.train_language
    # Setup adapters
    if adapter_args.train_adapter:
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )

        else:
            lang_adapter_name = None

        if adapter_args.load_region_adapter:
            region_adapter_config = AdapterConfig.load(
                adapter_args.region_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            region_adapter_name = model.load_adapter(
                adapter_args.load_region_adapter,
                config=region_adapter_config,
                load_as='region',
            )
        else:
            region_adapter_name = None

        if adapter_args.load_family_adapter:
            family_adapter_config = AdapterConfig.load(
                adapter_args.family_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            family_adapter_name = model.load_adapter(
                adapter_args.load_family_adapter,
                config=family_adapter_config,
                load_as='family',
            )
        else:
            family_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name and family_adapter_name is None and region_adapter_name is None:
            model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
            logger.info('set active {}-{}'.format(lang_adapter_name,task_name))
        elif lang_adapter_name and family_adapter_name and region_adapter_name is None:
            model.set_active_adapters(ac.Stack(family_adapter_name,lang_adapter_name, task_name))
            logger.info('set active {}-{}-{}'.format(family_adapter_name,lang_adapter_name,task_name))
        elif lang_adapter_name and family_adapter_name and region_adapter_name:
            model.set_active_adapters(ac.Stack(family_adapter_name,region_adapter_name, lang_adapter_name, task_name))
            logger.info('set active {}-{}-{}-{}'.format(family_adapter_name,region_adapter_name,lang_adapter_name,task_name))
        else:
            model.set_active_adapters(task_name)
            logger.info('set active {}'.format(task_name))
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Get the metric function
    metric = load_metric("xnli")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


    def load_tadapters(tad,tname):
                task_adapter_config = AdapterConfig.load(
                                config="pfeiffer", non_linearity="gelu", reduction_factor=16
                            )
                task_adapter_name = model.load_adapter(
                    tad,
                    config=task_adapter_config,
                    load_as=tname,
                )

                return task_adapter_name


    def load_ladapters(lad,lang):
        lang_adapter_config = AdapterConfig.load(
                        os.path.join(lad,'adapter_config.json'),
                        non_linearity="gelu",
                        reduction_factor=2
                    )

        lang_adapter_name = model.load_adapter(
            lad,
            config=lang_adapter_config,
            load_as=lang
        )
        return lang_adapter_name
    
    def load_fadapters(fadp,name):
        family_adapter_config = AdapterConfig.load(
                        os.path.join(fadp,'adapter_config.json'),
                        non_linearity="gelu",
                        reduction_factor=2
                    )

        family_adapter_name = model.load_adapter(
            fadp,
            config=family_adapter_config,
            load_as=name
        )
        return family_adapter_name


    def get_dataset(datalang):
        predict_dataset = load_dataset(data_args.dataset_name, datalang, split="test", cache_dir=model_args.cache_dir)
        label_list = predict_dataset.features["label"].names

        num_labels = len(label_list)

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

        return predict_dataset

    def predict_sing(trainer, predict_dataset,writer):
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=1)

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


        return metrics



    if training_args.do_predict_all:
        task_adapters =[
            'xnli'
        ]

        # lang_adapters={  
        #         "et_1m":"et_edt",
        #         "fi_1m":"fi_tdt",
        #         "hu_1m":"hu_szeged",
        #         "koi_10k":"koi_uh",
        #         "kpv_13k":"kpv_lattice",
        #         "krl_5k":"krl_kkpp",
        #         "mdf_5k":"mdf_jr",
        #         "myv_29k":"myv_jr",
        #         "olo_19k":"olo_kkpp",
        #         "sme_10k":"sme_giella",
        #         "sms_3k":"sms_giellagas" 
        # }

        # lang_adapters ={
        #     'bzd':'bzd',
        #     'oto':'oto',
        #     'nah':'nah',
        #     'tar':'tar',
        #     'hch':'hch',
        #     'aym':'aym',
        #     'cni':'cni',
        #     'gn':'gn',
        #     'quy':'quy',
        #     'shp':'shp'            
        # }

        # region_adapters = {
        #     'bzd':'chibchan',
        #     'oto':'utoaztacen',
        #     'nah':'utoaztacen',
        #     'tar':'utoaztacen',
        #     'hch':'utoaztacen',
        #     'aym':'quechuan',
        #     'cni':'quechuan',
        #     'gn':'quechuan',
        #     'quy':'quechuan',
        #     'shp':'quechuan' 
        # }
        # lang_adapters ={
        #     'nah':'nah',
        #     'tar':'tar',
        #     'hch':'hch',
        #     'gn':'gn',           
        # }
        # region_adapters={
        #     'nah':'aztecan',
        #     'tar':'tarahumaran',
        #     'hch':'corachol',
        #     'gn':'tupi_guarani'
        # }

        adapter_info={
            'nah':{
                'lang':'nah',
                'region':'aztecan',
                'lang_path_j':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/nah',
                'region_path':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/aztecan',
                'family_path_j':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/family',
                'lang_path_nj':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_nj40/nah/mlm',
                'family_path_nj':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_nj40/family/mlm',
                },
            'tar':{
                'lang':'tar',
                'region':'tarahumaran',
                'lang_path_j':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/tar',
                'region_path':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/aztecan',
                'family_path_j':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/family',
                'lang_path_nj':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_nj40/tar/mlm',
                'family_path_nj':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_nj40/family/mlm',
                },
            'hch':{
                'lang':'hch',
                'region':'corachol',
                'lang_path_j':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/hch',
                'region_path':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/aztecan',
                'family_path_j':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_40/family',
                'lang_path_nj':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_nj40/hch/mlm',
                'family_path_nj':'/scratch/ffaisal/run_mlm/adapter/uto_aztecan_nj40/family/mlm',
                },
            'gn':{
                'lang':'gn',
                'region':'tupi_guarani',
                'lang_path_j':'/scratch/ffaisal/run_mlm/adapter/tupian_40/gn',
                'region_path':'/scratch/ffaisal/run_mlm/adapter/tupian_40/tupi_guarani',
                'family_path_j':'/scratch/ffaisal/run_mlm/adapter/tupian_40/family',
                'lang_path_nj':'/scratch/ffaisal/run_mlm/adapter/tupian_nj40/gn/mlm',
                'family_path_nj':'/scratch/ffaisal/run_mlm/adapter/tupian_nj40/family/mlm',
                }
        }



        train_task_lang=data_args.task_name +"_" + model_args.train_language

        task_path='/scratch/ffaisal/run_mlm/experiments/xnli'
        # lang_path_nrg = '/scratch/ffaisal/run_mlm/adapter/ura_lang_joint'
        # family_path_nrg = '/scratch/ffaisal/run_mlm/adapter/ura_lang_joint'
        # lang_path_j = '/scratch/ffaisal/run_mlm/adapter/ame_lang_region_joint_r48'
        # family_path_j = '/scratch/ffaisal/run_mlm/adapter/ame_lang_region_joint_r48/family'
        # region_path = '/scratch/ffaisal/run_mlm/adapter/ame_lang_region_joint_r48'
        # lang_path_nj = '/scratch/ffaisal/run_mlm/adapter/ura_lang'
        # family_path_nj = '/scratch/ffaisal/run_mlm/adapter/ura_lang/family/mlm'




        logger.info("*** Predict ***")

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            writer = open(output_predictions_file, "w")



                

        # logger.info('\n\n\nexperiment 0: [task]')
        # tname=task_adapters[0]
        # tad=os.path.join(task_path,tname,train_task_lang)
        # logger.info(tad)
        # task_adapter_name=load_tadapters()
        # model.set_active_adapters(task_adapter_name)
        # logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # for lang in lang_adapters:
        #     logger.info(lang_adapters[lang])
        #     model.to(training_args.device)
        #     predict_dataset = get_dataset()
        #     metrics = predict_sing(trainer, predict_dataset,writer)
        #     logger.info("[task],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     writer.write("[task],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        # model.delete_adapter(task_adapter_name)


        # logger.info('\n\n\nexperiment 1: [task+lang->not joint]')
        # tname=task_adapters[5]
        # tad=os.path.join(task_path,tname,train_task_lang)
        # logger.info(tad)
        # task_adapter_name=load_tadapters()
        # logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # for lang in lang_adapters:
        #     lad= os.path.join(lang_path_nj,lang,'mlm')
        #     logger.info("lad: {}".format(lad))
        #     logger.info(lang_adapters[lang])
        #     lang_adapter_name=load_ladapters()          
        #     model.set_active_adapters(ac.Stack(lang_adapter_name, task_adapter_name))
        #     model.to(training_args.device)
        #     predict_dataset = get_dataset()
        #     metrics = predict_sing(trainer, predict_dataset,writer)
        #     model.delete_adapter(lang_adapter_name)
        #     logger.info("[lang+task],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     writer.write("[lang+task],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     model.delete_adapter(lang_adapter_name)
        # model.delete_adapter(task_adapter_name)

            
        # logger.info('\n\n\nexperiment 2: [lang+family->not joint]')
        # tname=task_adapters[6]
        # tad=os.path.join(task_path,tname,train_task_lang)
        # logger.info(tad)
        # task_adapter_name=load_tadapters()
        # fad = family_path_nj
        # family_adapter_name=load_fadapters(fad,'family')
        # logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # for lang in lang_adapters:
        #     lad= os.path.join(lang_path_nj,lang,'mlm')
        #     logger.info("lad: {}\nfad: {}".format(lad,fad))
        #     logger.info(lang_adapters[lang])
        #     lang_adapter_name=load_ladapters()          
        #     model.set_active_adapters(ac.Stack(family_adapter_name,lang_adapter_name, task_adapter_name))
        #     model.to(training_args.device)
        #     predict_dataset = get_dataset()
        #     metrics = predict_sing(trainer, predict_dataset,writer)
        #     logger.info("[family+lang+task],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     writer.write("[family+lang+task],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     model.delete_adapter(lang_adapter_name)
        # model.delete_adapter(family_adapter_name)
        # model.delete_adapter(task_adapter_name)

            
        # logger.info('\n\n\nexperiment 3: [task+lang->joint]')
        # tname=task_adapters[0]
        # tad=os.path.join(task_path,tname,train_task_lang)
        # logger.info(tad)
        # task_adapter_name=load_tadapters()
        # logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # for lang in lang_adapters:
        #     lad= os.path.join(lang_path_j,lang)
        #     logger.info("lad: {}".format(lad))
        #     logger.info(lang_adapters[lang])
        #     lang_adapter_name=load_ladapters()          
        #     model.set_active_adapters(ac.Stack(lang_adapter_name, task_adapter_name))
        #     model.to(training_args.device)
        #     predict_dataset = get_dataset()
        #     metrics = predict_sing(trainer, predict_dataset,writer)
        #     logger.info("[lang+task->joint],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     writer.write("[lang+task->joint],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     model.delete_adapter(lang_adapter_name)
        # model.delete_adapter(task_adapter_name)
            
        # logger.info('\n\n\nexperiment 4: [lang+family->joint]')
        # tname=task_adapters[0]
        # tad=os.path.join(task_path,tname,train_task_lang)
        # logger.info(tad)
        # task_adapter_name=load_tadapters()
        # fad = family_path_j
        # family_adapter_name=load_fadapters(fad, 'family')
        # logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # for lang in lang_adapters:
        #     lad= os.path.join(lang_path_j,lang)
        #     logger.info("lad: {}\nfad: {}".format(lad,fad))
        #     logger.info(lang_adapters[lang])
        #     lang_adapter_name=load_ladapters()          
        #     model.set_active_adapters(ac.Stack(family_adapter_name,lang_adapter_name, task_adapter_name))
        #     model.to(training_args.device)
        #     predict_dataset = get_dataset()
        #     metrics = predict_sing(trainer, predict_dataset,writer)
        #     logger.info("[family+lang+task->joint],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     writer.write("[family+lang+task->joint],%s,%s,%s,%s\n" % (tname, lang, metrics['predict_accuracy'], metrics['predict_accuracy']))
        #     model.delete_adapter(lang_adapter_name)
        # model.delete_adapter(family_adapter_name)
        # model.delete_adapter(task_adapter_name)

        def  do_prediction_all(text,adapter_list):
            logger.info('\n\n\n{}'.format(text))
            model.set_active_adapters(adapter_list)
            # else:
            #     model.set_active_adapters(ac.Stack(adapter_list))
            model.to(training_args.device)
            metrics = predict_sing(trainer, predict_dataset,writer)
            logger.info("%s,%s,%s,%s,%s\n" % (text,tname, 
                lang, 
                metrics['predict_accuracy'], 
                metrics['predict_accuracy']))
            writer.write("%s,%s,%s,%s,%s\n" % (text,tname, 
                lang, 
                metrics['predict_accuracy'], 
                metrics['predict_accuracy']))
            model.set_active_adapters(None)

        tname=task_adapters[0]
        tad=os.path.join(task_path,tname,train_task_lang)
        logger.info('tad:{}, task_path:{}'.format(tad, tname))
        task_adapter_name=load_tadapters(tad, tname)
        for lang in adapter_info:
            print(lang)
            predict_dataset = get_dataset(adapter_info[lang]['lang'])
            lad = adapter_info[lang]['lang_path_j']
            lang_adapter_name=load_ladapters(lad,lang)
            rad = adapter_info[lang]['region_path']
            region_adapter_name=load_fadapters(rad,'region') 
            fad = adapter_info[lang]['family_path_j']
            family_adapter_name=load_fadapters(fad, 'family')

            do_prediction_all('[task]',[task_adapter_name])           
            do_prediction_all('[task+lang->joint]',[lang_adapter_name, task_adapter_name])
            do_prediction_all('[task+lang+family->joint]',[family_adapter_name,lang_adapter_name, task_adapter_name])
            do_prediction_all('[task+lang+region+family->joint]',[family_adapter_name,region_adapter_name,
                                               lang_adapter_name, task_adapter_name])

            model.delete_adapter(lang_adapter_name)
            model.delete_adapter(region_adapter_name)
            model.delete_adapter(family_adapter_name)

            lad = adapter_info[lang]['lang_path_nj']
            lang_adapter_name=load_ladapters(lad,lang) 
            fad = adapter_info[lang]['family_path_nj']
            family_adapter_name=load_fadapters(fad, 'family')
            do_prediction_all('[task+lang]',[lang_adapter_name, task_adapter_name])
            do_prediction_all('[task+lang+family]',[family_adapter_name,lang_adapter_name, task_adapter_name])
        model.delete_adapter(task_adapter_name)



            

        # logger.info('\n\n\nexperiment 5: [lang+region+family->joint]')
        # tname=task_adapters[0]
        # tad=os.path.join(task_path,tname,train_task_lang)
        # logger.info('tad:{}, task_path:{}'.format(tad, tname))
        # task_adapter_name=load_tadapters()
        # fad = os.path.join(region_path,'family')
        # family_adapter_name=load_fadapters(fad, 'family')
        # logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # for lang in lang_adapters:
        #     lad= os.path.join(region_path,lang)
        #     logger.info("lad: {}\nfad: {}".format(lad,fad))
        #     logger.info(lang_adapters[lang])
        #     lang_adapter_name=load_ladapters() 
            
        #     rad= os.path.join(region_path,region_adapters[lang])
        #     logger.info(rad)
        #     region_adapter_name=load_fadapters(rad,'region') 
            
        #     model.set_active_adapters(ac.Stack(family_adapter_name,region_adapter_name,
        #                                        lang_adapter_name, task_adapter_name))
        #     model.to(training_args.device)
        #     predict_dataset = get_dataset()
        #     metrics = predict_sing(trainer, predict_dataset,writer)
        #     logger.info("[family+region+lang+task->joint],%s,%s,%s,%s\n" % (tname, 
        #         lang, 
        #         metrics['predict_accuracy'], 
        #         metrics['predict_accuracy']))
        #     writer.write("[family+region+lang+task->joint],%s,%s,%s,%s\n" % (tname, 
        #         lang, 
        #         metrics['predict_accuracy'], 
        #         metrics['predict_accuracy']))
        #     model.delete_adapter(region_adapter_name)
        #     model.delete_adapter(lang_adapter_name)
        # model.delete_adapter(family_adapter_name)
        # model.delete_adapter(task_adapter_name)


        # logger.info('\n\n\nexperiment 6: [lang+family->joint (ntrj)]')
        # tname=task_adapters[4]
        # tad=os.path.join(task_path,tname,train_task_lang)
        # logger.info(tad)
        # task_adapter_name=load_tadapters()
        # fad = os.path.join(family_path_nrg,'family')
        # family_adapter_name=load_fadapters(fad, 'family')
        # logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # for lang in lang_adapters:
        #     lad= os.path.join(lang_path_nrg,lang)
        #     logger.info("lad: {}\nfad: {}".format(lad,fad))
        #     logger.info(lang_adapters[lang])
        #     lang_adapter_name=load_ladapters()          
        #     model.set_active_adapters(ac.Stack(family_adapter_name,lang_adapter_name, task_adapter_name))
        #     model.to(training_args.device)
        #     predict_dataset = get_dataset()
        #     metrics = predict_sing(trainer, predict_dataset,writer)
        #     logger.info("[family+lang+task->joint_ntrg],%s,%s,%s,%s\n" % (tname, 
        #         lang, 
        #         metrics['predict_accuracy'], 
        #         metrics['predict_accuracy']))
        #     writer.write("[family+lang+task->joint_ntrg],%s,%s,%s,%s\n" % (tname, 
        #         lang, 
        #         metrics['predict_accuracy'], 
        #         metrics['predict_accuracy']))
        #     model.delete_adapter(lang_adapter_name)
        # model.delete_adapter(family_adapter_name)
        # model.delete_adapter(task_adapter_name)
           
        
        logger.info('-------------')

if __name__ == "__main__":
    main()
