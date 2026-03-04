from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from model import peft_model, tokenizer
from dataset import tokenized_dataset

# Define collator for data merge in sequence-to-sequence outputs
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=peft_model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

import os

# Define output directory
output_dir = "./lora-flan-t5-dolly"

# Define Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    per_device_train_batch_size=2, # Micro-batching
    gradient_accumulation_steps=8, # Simulates a batch size of 16
    learning_rate=2e-4, # Standard LoRA learning rate
    num_train_epochs=1, # Single pass for architectural validation
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch",
    report_to="none", # Disable WandB/external logging for this local test
)

# Compile the Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Initialize model training
print("Initializing PEFT Matrix Multiplication...")
trainer.train()


# Save localized weights
print(f"Exporting LoRA adapters to {output_dir}...")
peft_model.save_pretrained(output_dir)
print("✅ Compute cycle complete. Architecture validated.")
