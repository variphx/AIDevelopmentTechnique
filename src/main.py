from schema.dataset import RawTrafficRulesDataset
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
argparser.add_argument("--epochs", type=int)
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
    per_device_train_batch_size=64,
    num_train_epochs=argv.epochs,
    learning_rate=1e-4,
    lr_scheduler_type=transformers.SchedulerType.COSINE,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=8,
    log_level="debug",
    report_to="none",
    neftune_noise_alpha=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    callbacks=[],
)

trainer.train()
trainer.save_model(argv.output_dir)
