from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_id = "google/flan-t5-base"

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    target_modules=["q", "v"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

peft_model = get_peft_model(original_model, config)

peft_model.print_trainable_parameters()
