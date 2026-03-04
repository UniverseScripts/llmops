from datasets import load_dataset
from model import tokenizer

# 1. Ingest the dataset (Extracting a 5% validation slice for local compute)
print("Downloading enterprise production dataset...")
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:5%]")

# 2. Architect the Tokenization Pipeline
def preprocess_function(sample):
    
    prompts = [
        f"Instruction: {inst}\nContext: {ctx}\nResponse:" 
        for inst, ctx in zip(sample['instruction'], sample['context'])
    ]
    
    model_inputs = tokenizer(
        prompts,
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    
    labels = tokenizer(
        sample["response"],
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing data...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

print(f"✅ Tokenized dataset size: {len(tokenized_dataset)} samples")