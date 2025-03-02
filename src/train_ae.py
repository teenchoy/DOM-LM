import argparse
import os
import sys
from pathlib import Path
from multiprocessing import freeze_support
from utils import label2id, id2label

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from collections import OrderedDict

import src.domlm as model 
import src.dataset as dataset
from src.data_collator_ae import DataCollatorForAttributeExtraction


def train(pretrained_path, ds_path, output_dir, per_device_train_batch_size, gradient_accumulation_steps, dataloader_num_workers, epochs, report_to):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta = AutoModel.from_pretrained("roberta-base")
    roberta_config = roberta.config


    roberta_config_dict = roberta_config.to_dict()
    roberta_config_dict["_name_or_path"] = "domlm"
    roberta_config_dict["architectures"] = ["DOMLMForMaskedLM"]
    domlm_config = model.DOMLMConfig.from_dict(roberta_config_dict)
    domlm_config.num_labels = len(label2id.keys()) + 1
    # domlm_config.save_pretrained("../domlm-config/")
    # domlm = model.DOMLMForTokenClassification(domlm_config)
    domlm = model.DOMLMForTokenClassification.from_pretrained(pretrained_path, config=domlm_config)

    state_dict = OrderedDict((f"domlm.{k}",v) for k,v in roberta.state_dict().items())
    domlm.load_state_dict(state_dict,strict=False)

    # dataset_path = ROOT / "data/swde_preprocessed"
    dataset_path = Path(ds_path)

    print(f"Loading datasets from {dataset_path}...")
    train_ds = dataset.SWDEDataset(dataset_path, domain="movie")
    test_ds = dataset.SWDEDataset(dataset_path, domain="movie", split="test")

    # tokenizer.pad_token = tokenizer.eos_token # why do we need this?
    data_collator = DataCollatorForAttributeExtraction(tokenizer=tokenizer, mlm_probability=0.15)

    # install apex:
    # comment lines 32-40 in apex/setup.py
    # pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

    #TODO: add evaluation metrics (ppl, etc.)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        # optim="adamw_apex_fused", # only with apex installed
        weight_decay=0.01,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # gradient_checkpointing=True, # vram is enough without checkpointing on A4000
        bf16 = True, # If not Ampere: fp16 = True
        # tf32 = True, # Ampere Only
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        # report_to=None
    )

    trainer = Trainer(
        model=domlm,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='/Users/zehengxiao/Downloads/checkpoint-2955', help='preprocess data for tasks')
    parser.add_argument('--ds_path', type=str, default='/Users/zehengxiao/Projects/Python/DOM-LM/domlm/data/mini', help='data directory')
    parser.add_argument('--config', type=str, default='domlm-config/config.json', help='config file')
    parser.add_argument('--output_dir', type=str, default='results_ae', help='output directory')
    parser.add_argument('--per_device_train_batch_size', type=int, default='16', help='output directory')
    parser.add_argument('--gradient_accumulation_steps', type=int, default='4', help='output directory')
    parser.add_argument('--dataloader_num_workers', type=int, default='8', help='output directory')
    parser.add_argument('--epochs', type=float, default='5', help='output directory')
    parser.add_argument('--report_to', type=bool, default='True', help='output directory')
    args = parser.parse_args()

    pretrained_path = args.pretrained_path
    ds_path = args.ds_path
    config = args.config
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    dataloader_num_workers = args.dataloader_num_workers
    epochs = args.epochs
    output_dir = args.output_dir
    report_to = args.report_to

    train(pretrained_path, ds_path, output_dir, per_device_train_batch_size, gradient_accumulation_steps, dataloader_num_workers, epochs, report_to)
    # trainer.train(resume_from_checkpoint=False, input_dir)

