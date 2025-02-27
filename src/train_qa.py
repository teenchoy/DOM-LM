import argparse
import os
import sys
from pathlib import Path
from multiprocessing import freeze_support

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from collections import OrderedDict

import src.domlm as model 
import src.dataset_qa as dataset
from src.data_collator_qa import DataCollatorForDOMNodeMask
from src.utils import label2id, id2label


def train(pretrained_path, ds_path, config, output_dir):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta = AutoModel.from_pretrained("roberta-base")
    roberta_config = roberta.config


    roberta_config_dict = roberta_config.to_dict()
    roberta_config_dict["_name_or_path"] = "domlm"
    roberta_config_dict["architectures"] = ["DOMLMForMaskedLM"]
    domlm_config = model.DOMLMConfig.from_dict(roberta_config_dict)
    # domlm_config.save_pretrained("../domlm-config/")
    domlm_config.num_labels = 2

    # domlm = model.DOMLMForSequenceClassification(domlm_config)
    domlm = model.DOMLMForQuestionAnswering.from_pretrained(pretrained_path, config=domlm_config)
    # domlm = model.DOMLMForQuestionAnswering.from_pretrained("/content/drive/MyDrive/domlm_results/checkpoint-2955", config=domlm_config)

    state_dict = OrderedDict((f"domlm.{k}",v) for k,v in roberta.state_dict().items())
    domlm.load_state_dict(state_dict,strict=False)

    dataset_path = Path(ds_path)
    # dataset_path = Path("/content/drive/MyDrive/colab/pkls/output")
    print(f"Loading datasets from {dataset_path}...")
    train_ds = dataset.WebSRCDataset(dataset_path)
    test_ds = dataset.WebSRCDataset(dataset_path,split="test")

    # tokenizer.pad_token = tokenizer.eos_token # why do we need this?
    data_collator = DataCollatorForDOMNodeMask(tokenizer=tokenizer, mlm_probability=0.15)

    # install apex:
    # comment lines 32-40 in apex/setup.py
    # pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

    #TODO: add evaluation metrics (ppl, etc.)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        # optim="adamw_apex_fused", # only with apex installed
        weight_decay=0.01,
        num_train_epochs=5,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        save_steps=20,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        gradient_accumulation_steps=3,
        # gradient_checkpointing=True, # vram is enough without checkpointing on A4000
        # bf16 = True, # If not Ampere: fp16 = True
        # tf32 = True, # Ampere Only
        dataloader_num_workers=3,
        dataloader_pin_memory=True,
        save_total_limit=1
    )

    trainer = CustomTrainer(
        model=domlm,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=False)

class CustomTrainer(Trainer):
    def training_step(self, model, inputs, return_outputs=False):
        model.train()
        inputs = self._prepare_inputs(inputs)  # Move inputs to device

        # Forward pass
        outputs = model(**inputs)  # This is where forward() is called
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        print("Forward inputs:", {k: v.shape for k, v in inputs.items()})  # Debugging
        print("Loss:", loss.item())

        return loss



if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='', help='preprocess data for tasks')
    parser.add_argument('--ds_path', type=str, default='', help='data directory')
    parser.add_argument('--config', type=str, default='domlm-config/config.json', help='config file')
    parser.add_argument('--output_dir', type=str, default='data/qa_preprocessed', help='output directory')
    args = parser.parse_args()

    pretrained_path = args.pretrained_path
    ds_path = args.ds_path
    config = args.config
    output_dir = args.output_dir

    train(pretrained_path, ds_path, config, output_dir)
    # trainer.train(resume_from_checkpoint=False, input_dir)

