from schema.dataset import RawTrafficRulesDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import transformers
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--data-dir", type=str)
argparser.add_argument("--base-model", type=str)
argparser.add_argument("--output-dir", type=str)
argv = argparser.parse_args()

base_model = AutoModelForMaskedLM.from_pretrained(argv.base_model)
tokenizer = AutoTokenizer.from_pretrained(argv.base_model)

train_dataset = RawTrafficRulesDataset(
    "data/Main-QCVN 41_2019-BGTVT.txt",
    chunk_size=200,
    tokenizer=tokenizer,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
)

lora_config = LoraConfig(
    init_lora_weights="olora",
    task_type="MASKED_LM",
    target_modules=[
        "query",
        "value",
    ],
)

model = get_peft_model(base_model, lora_config)

training_args = TrainingArguments(
    output_dir=argv.output_dir,
    overwrite_output_dir=True,
    do_train=True,
    per_device_train_batch_size=128,
    num_train_epochs=15,
    lr_scheduler_type=transformers.SchedulerType.COSINE,
    logging_steps=0.1,
    metric_for_best_model="loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    callbacks=[
        transformers.PrinterCallback(),
        transformers.EarlyStoppingCallback(early_stopping_threshold=1e-4),
    ],
)

trainer.train()
trainer.save_model(argv.output_dir)
